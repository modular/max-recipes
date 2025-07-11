import os
from io import BytesIO
import base64
import tempfile
import logging

import torch
import fitz  # PyMuPDF
import gradio as gr
from PIL import Image
from tqdm import tqdm
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    retry_if_result,
)
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from colpali_engine.models import ColPali, ColPaliProcessor
from rerankers import Reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8010/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local")
LLM_HEALTH_URL = os.getenv("LLM_HEALTH_URL", "http://localhost:8010/v1/health")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_HEALTH_URL = os.getenv("QDRANT_HEALTH_URL", "http://localhost:6333/healthz")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "vidore/colpali-v1.3")
COLLECTION_NAME = "pdf_images"
VECTOR_DIM = 128
BATCH_SIZE = 8
DEVICE = "cuda"
SYSTEM_PROMPT = """
You are a helpful document-answering assistant that answers questions about the PDF document content.
When responding to queries, consider the context and intent of the question.
Do not hallucinate. Be concise and to the point. If you don't know the answer, say so.
"""
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

client = OpenAI(base_url=LLM_SERVER_URL, api_key=LLM_API_KEY)

def wait_for_healthy(base_url: str, service_name: str, health_url: str):
    """Wait for a service to be healthy with configurable retry settings."""
    @retry(
        stop=stop_after_attempt(60),
        wait=wait_fixed(20),
        retry=(
            retry_if_exception_type(requests.RequestException)
            | retry_if_result(lambda r: r.status_code != 200)
        ),
        before_sleep=lambda retry_state: logger.info(
            f"Waiting for {service_name} at {health_url} to start (attempt {retry_state.attempt_number}/60)..."
        ),
    )
    def _check_health():
        return requests.get(health_url, timeout=5)

    return _check_health()

class PDFProcessor:
    def __init__(self, temp_dir="./temp_images"):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    def extract_images(self, pdf_path):
        """Extract images from PDF file"""
        images = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img = Image.open(BytesIO(pix.tobytes()))
            images.append(img)

        return images

    def process_pdf(self, pdf_file):
        """Process uploaded PDF file and extract images"""
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

        try:
            content = pdf_file.read()
        except AttributeError:
            if hasattr(pdf_file, 'name'):
                with open(pdf_file.name, 'rb') as f:
                    content = f.read()
            else:
                content = str(pdf_file).encode('utf-8')

        temp_pdf.write(content)
        temp_pdf.close()

        images = self.extract_images(temp_pdf.name)
        os.unlink(temp_pdf.name)

        return images

class EmbedData:
    def __init__(self, embed_model_name=EMBEDDING_MODEL, batch_size=BATCH_SIZE):
        self.embed_model_name = embed_model_name
        self.batch_size = batch_size
        self.embeddings = []
        logger.info(f"Loading ColPali model from {embed_model_name}...")
        self.embed_model, self.processor = self._load_embed_model()
        logger.info("ColPali model loaded successfully")

    def _load_embed_model(self):
        embed_model = ColPali.from_pretrained(
            self.embed_model_name,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            trust_remote_code=True
        )

        processor = ColPaliProcessor.from_pretrained(self.embed_model_name)
        return embed_model, processor

    def get_query_embedding(self, query):
        with torch.amp.autocast(device_type='cuda'), torch.no_grad():
            query = self.processor.process_queries([query]).to(self.embed_model.device)
            query_embedding = self.embed_model(**query)

        return query_embedding[0].cpu().float().numpy().tolist()

    def generate_embedding(self, images):
        with torch.amp.autocast(device_type='cuda'), torch.no_grad():
            batch_images = self.processor.process_images(images).to(self.embed_model.device)
            image_embeddings = self.embed_model(**batch_images).cpu().float().numpy().tolist()

        torch.cuda.empty_cache()
        return image_embeddings

    def batch_iterate(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i+batch_size]

    def embed(self, images):
        self.images = images
        self.embeddings = []

        for batch_images in tqdm(self.batch_iterate(images, self.batch_size), desc="Generating embeddings"):
            batch_embeddings = self.generate_embedding(batch_images)
            self.embeddings.extend(batch_embeddings)

        return self.embeddings

class QdrantVectorDB:
    def __init__(self, collection_name, vector_dim=128, batch_size=4):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        logger.info("Connecting to Qdrant...")

        self.client = QdrantClient(
            url=QDRANT_URL,
            prefer_grpc=True
        )
        logger.info("Connected to Qdrant successfully")

    def create_collection(self):
        if self.client.collection_exists(collection_name=self.collection_name):
            logger.info(f"Deleting existing collection {self.collection_name} to recreate with proper configuration...")
            self.client.delete_collection(collection_name=self.collection_name)

        logger.info(f"Creating collection {self.collection_name}...")
        self.client.create_collection(
            collection_name=self.collection_name,
            on_disk_payload=True,
            vectors_config=models.VectorParams(
                size=self.vector_dim,
                distance=models.Distance.COSINE,
                on_disk=True,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )
        logger.info(f"Collection {self.collection_name} created successfully")

    def image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def ingest_data(self, images, embeddings):
        for i in range(0, len(embeddings), self.batch_size):
            batch_embeddings = embeddings[i:i+self.batch_size]
            batch_images = images[i:i+self.batch_size]

            points = []
            for j, embedding in enumerate(batch_embeddings):
                image_b64 = self.image_to_base64(batch_images[j])

                point = models.PointStruct(
                    id=i+j,
                    vector=embedding,
                    payload={"image": image_b64}
                )

                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

        logger.info("Data ingestion complete")
        return len(embeddings)

class Retriever:
    def __init__(self, vector_db, embed_data, use_reranker=True):
        self.vector_db = vector_db
        self.embed_data = embed_data
        self.use_reranker = use_reranker

        if self.use_reranker:
            logger.info("Initializing reranker...")
            self.reranker = Reranker('cross-encoder')
            logger.info("Reranker initialized successfully")

    def search(self, query, display_limit=3, initial_fetch=10):
        logger.info(f"Searching for: '{query}'")
        query_embedding = self.embed_data.get_query_embedding(query)

        is_author_query = any(keyword in query.lower() for keyword in ["author", "who wrote", "written by"])
        is_metadata_query = is_author_query or any(keyword in query.lower() for keyword in ["publisher", "published", "edition", "year"])

        if is_metadata_query:
            initial_fetch = 15
            display_limit = 2

        results = self.vector_db.client.query_points(
            collection_name=self.vector_db.collection_name,
            query=query_embedding,
            limit=initial_fetch
        )

        filtered_results = []
        for result in results.points:
            page_content = result.payload.get("text", "").lower()

            if is_metadata_query:
                priority_keywords = ['title page', 'copyright page', 'about the author', 'author', 'published by']
                if any(keyword in page_content for keyword in priority_keywords):
                    filtered_results.append(result)

        for result in results.points:
            if result not in filtered_results:
                filtered_results.append(result)

        if self.use_reranker and filtered_results and not is_metadata_query:
            try:
                docs = []
                doc_ids = []
                for result in filtered_results:
                    docs.append(f"Page {result.id}")
                    doc_ids.append(result.id)

                reranked = self.reranker.rank(
                    query=query,
                    docs=docs,
                    doc_ids=doc_ids
                )

                top_results = reranked.top_k(display_limit)

                final_results = []
                for reranked_result in top_results:
                    for orig_result in filtered_results:
                        if orig_result.id == reranked_result.doc_id:
                            final_results.append(orig_result)
                            break

                logger.info(f"Reranking successful, returned {len(final_results)} results")
                return models.QueryResponse(points=final_results)

            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using original order.")
                return models.QueryResponse(points=filtered_results[:display_limit])

        return models.QueryResponse(points=filtered_results[:display_limit])

class RAG:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm_client = client

    def generate_context(self, query, display_limit=5, llm_limit=1):
        """Retrieve relevant images for the query"""
        results = self.retriever.search(query, display_limit=display_limit)

        context_images = []
        page_info = []
        for result in results.points:
            image_b64 = result.payload.get("image")
            if image_b64:
                context_images.append(image_b64)
                page_info.append(f"Page {result.id}")

        logger.info(f"Retrieved {len(context_images)} context images")
        return context_images, page_info, llm_limit

    def query(self, query):
        """Generate response using RAG with Llama 3.2 Vision"""
        context_images, page_info, llm_limit = self.generate_context(query)

        if not context_images:
            return "I couldn't find relevant information in the document to answer your question.", []

        display_images = []
        llm_images = []

        for i, img_b64 in enumerate(context_images):
            try:
                img_data = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_data))

                display_images.append(img_b64)

                if i < llm_limit:
                    img_small = img.resize((256, 256), Image.LANCZOS)
                    buffered = BytesIO()
                    img_small.save(buffered, format="JPEG", quality=70)
                    llm_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
            except Exception as e:
                logger.warning(f"Error processing image: {e}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": query}
            ]}
        ]

        for img_b64 in llm_images:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            })

        logger.info("Generating response with Llama 3.2 Vision...")
        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.3
        )

        return response.choices[0].message.content, page_info

class UI:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embed_data = EmbedData(batch_size=BATCH_SIZE)
        self.vector_db = QdrantVectorDB(
            collection_name=COLLECTION_NAME,
            vector_dim=VECTOR_DIM,
            batch_size=BATCH_SIZE
        )
        self.vector_db.create_collection()
        self.retriever = Retriever(self.vector_db, self.embed_data)
        self.rag = RAG(self.retriever)

        self.processed_images = []
        self.chat_history = []
        self.current_context_images = []
        self.current_page_info = []

    def process_pdf(self, pdf_file):
        """Process PDF and store embeddings"""
        if pdf_file is None:
            return "Please upload a PDF file."

        logger.info("Processing PDF file...")
        self.processed_images = self.pdf_processor.process_pdf(pdf_file)

        if not self.processed_images:
            return "No images found in the PDF."

        logger.info(f"Generating embeddings for {len(self.processed_images)} images...")
        embeddings = self.embed_data.embed(self.processed_images)

        count = self.vector_db.ingest_data(self.processed_images, embeddings)

        return f"Processed {count} images from the PDF. You can now ask questions!"

    def chat(self, message, history):
        """Handle chat messages with the messages format"""
        if not self.processed_images:
            return "Please upload a PDF first.", history, None

        response, page_info = self.rag.query(message)
        self.current_page_info = page_info
        context_images = []
        for i, page in enumerate(page_info):
            try:
                page_idx = int(page.replace("Page ", ""))
                if 0 <= page_idx < len(self.processed_images):
                    context_images.append(self.processed_images[page_idx])
            except Exception as e:
                logger.warning(f"Error processing image: {e}")
                pass

        self.current_context_images = context_images
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        return "", history, context_images

    def launch(self):
        """Launch the Gradio interface"""
        custom_css = """
            .gradio-container {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            .gr-button {
                background-color: #1f6feb !important;
                border: none !important;
            }
            .gr-button:hover {
                background-color: #388bfd !important;
            }
            h1 {
                text-align: center !important;
                color: #1f6feb !important;
            }
            h3 {
                text-align: center;
                color: #666;
            }
        """

        with gr.Blocks(title="Multi-Modal PDF RAG", theme=gr.themes.Soft(), css=custom_css) as demo:
            gr.Markdown(
                """
                # Multi-Modal PDF RAG with ColPali, Llama3.2-Vision, Qdrant, Reranker powered by [MAX](https://docs.modular.com/max/serve) 🚀
                ### A powerful document question-answering system combining state-of-the-art visual and language models
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="Upload PDF")
                    process_btn = gr.Button("Process PDF")
                    status_output = gr.Textbox(label="Status")

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(type='messages')
                    msg_input = gr.Textbox(label="Ask a question about the PDF", placeholder="What's in this document?")
                    clear_btn = gr.Button("Clear Chat")

                    context_gallery = gr.Gallery(label="Context Images", show_label=True, elem_id="gallery", columns=2, height="auto")

            process_btn.click(
                fn=self.process_pdf,
                inputs=[pdf_input],
                outputs=[status_output]
            )

            msg_input.submit(
                fn=self.chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, context_gallery]
            )

            clear_btn.click(
                lambda: (None, [], None),
                None,
                [msg_input, chatbot, context_gallery],
                queue=False
            )

        demo.launch(share=False)

def main():
    logger.info("Checking if MAX is healthy...")
    wait_for_healthy(LLM_SERVER_URL, "MAX", LLM_HEALTH_URL)
    logger.info("MAX is healthy")

    logger.info("Checking if Qdrant is healthy...")
    wait_for_healthy(QDRANT_URL, "Qdrant", QDRANT_HEALTH_URL)
    logger.info("Qdrant is healthy")

    logger.info("Starting Gradio UI...")
    ui = UI()
    ui.launch()

if __name__ == "__main__":
    main()
