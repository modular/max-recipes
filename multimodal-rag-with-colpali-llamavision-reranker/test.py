#!/usr/bin/env python3
"""
Integration test for multimodal-rag-with-colpali-llamavision-reranker recipe.
Tests that the recipe can start and the endpoints are accessible.
"""

import os
import subprocess
import time
import signal
import sys
from pathlib import Path

# Optional imports with fallbacks
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not available, skipping HTTP tests")

# Try to check for CUDA accelerator availability
try:
    import max.driver
    HAS_MAX = True
    # Use max.driver.accelerator_api and accelerator_count to detect CUDA specifically
    try:
        accelerator_count = max.driver.accelerator_count()
        accelerator_api = max.driver.accelerator_api()
        HAS_CUDA = accelerator_count > 0 and accelerator_api == "cuda"
        if HAS_CUDA:
            print(f"Detected {accelerator_count} CUDA accelerator(s)")
        elif accelerator_count > 0:
            print(f"Detected {accelerator_count} {accelerator_api} accelerator(s), but this recipe requires CUDA")
    except:
        HAS_CUDA = False
except ImportError:
    HAS_MAX = False
    HAS_CUDA = False


class MultimodalRAGIntegrationTest:
    """Integration test for multimodal-rag-with-colpali-llamavision-reranker recipe."""
    
    def __init__(self):
        self.recipe_dir = Path(__file__).parent
        self.process = None
        self.llm_endpoint = "http://localhost:8010/v1/health"
        self.qdrant_endpoint = "http://localhost:6333/healthz"
        self.app_endpoint = "http://localhost:7860"  # Default Gradio port
        self.startup_timeout = 300  # 5 minutes for startup
        self.check_interval = 5  # Check every 5 seconds
    
    def start_services(self):
        """Start the multimodal-rag services."""
        print("Starting multimodal-rag services...")
        
        # Change to recipe directory
        os.chdir(self.recipe_dir)
        
        # Start the services using pixi run app
        try:
            self.process = subprocess.Popen(
                ["pixi", "run", "app"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid  # Create new process group
            )
            print(f"Started process with PID: {self.process.pid}")
            return True
        except Exception as e:
            print(f"Failed to start services: {e}")
            return False
    
    def stop_services(self):
        """Stop the services."""
        if self.process:
            print("Stopping services...")
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                # If graceful shutdown fails, force kill
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process = None
            print("Services stopped")
    
    def check_endpoint(self, url, service_name):
        """Check if an endpoint is responding."""
        if not HAS_REQUESTS:
            print(f"Skipping {service_name} endpoint check (requests not available)")
            return False
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is responding at {url}")
                return True
            else:
                print(f"❌ {service_name} returned status {response.status_code} at {url}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ {service_name} not responding at {url}: {e}")
            return False
    
    def check_docker_available(self):
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ["docker", "ps"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                print("✅ Docker is available and running")
                return True
            else:
                print(f"❌ Docker is not running: {result.stderr}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"❌ Docker not available: {e}")
            return False
    
    def check_cuda_available(self):
        """Check if CUDA accelerators are available using max.driver API."""
        if not HAS_MAX:
            print("❌ MAX package not available")
            return False
        
        try:
            accelerator_count = max.driver.accelerator_count()
            accelerator_api = max.driver.accelerator_api()
            
            if accelerator_count > 0 and accelerator_api == "cuda":
                print(f"✅ Found {accelerator_count} CUDA accelerator(s)")
                return True
            elif accelerator_count > 0:
                print(f"❌ Found {accelerator_count} {accelerator_api} accelerator(s), but this recipe requires CUDA")
                return False
            else:
                print("❌ No accelerators detected")
                return False
        except Exception as e:
            print(f"❌ Error checking CUDA availability: {e}")
            return False
    
    def wait_for_services(self):
        """Wait for all services to start and be ready."""
        print("Waiting for services to start...")
        start_time = time.time()
        llm_ready = False
        qdrant_ready = False
        app_ready = False
        
        while time.time() - start_time < self.startup_timeout:
            # Check if process is still running
            if self.process and self.process.poll() is not None:
                print("Process exited unexpectedly")
                return False
            
            # Check LLM endpoint
            if not llm_ready:
                llm_ready = self.check_endpoint(self.llm_endpoint, "MAX LLM Server")
            
            # Check Qdrant endpoint
            if not qdrant_ready:
                qdrant_ready = self.check_endpoint(self.qdrant_endpoint, "Qdrant")
            
            # Check Gradio app endpoint
            if not app_ready:
                app_ready = self.check_endpoint(self.app_endpoint, "Gradio App")
            
            # If all are ready, we're done
            if llm_ready and qdrant_ready and app_ready:
                print("✅ All services are ready!")
                return True
            
            # Wait before next check
            time.sleep(self.check_interval)
            elapsed = time.time() - start_time
            print(f"Elapsed time: {elapsed:.1f}s / {self.startup_timeout}s")
        
        print(f"❌ Services did not start within {self.startup_timeout} seconds")
        return False
    
    def test_basic_configuration(self):
        """Test basic configuration files exist."""
        print("Testing basic configuration...")
        
        # Check essential files exist
        essential_files = [
            "pyproject.toml",
            "metadata.yaml",
            "app.py",
            "Procfile",
            "Procfile.clean"
        ]
        
        for file_name in essential_files:
            file_path = self.recipe_dir / file_name
            if not file_path.exists():
                print(f"❌ Missing essential file: {file_name}")
                return False
            print(f"✅ Found {file_name}")
        
        # Check pyproject.toml has pixi configuration
        pyproject_path = self.recipe_dir / "pyproject.toml"
        with open(pyproject_path) as f:
            content = f.read()
            if "[tool.pixi" not in content:
                print("❌ pyproject.toml missing pixi configuration")
                return False
            print("✅ pyproject.toml has pixi configuration")
        
        # Test that it references pixi, not magic
        if "magic" in content.lower():
            print("❌ Found 'magic' references in pyproject.toml")
            return False
        print("✅ No 'magic' references found")
        
        # Test that it uses modular package
        if "modular" not in content:
            print("❌ Missing 'modular' package reference")
            return False
        print("✅ Found 'modular' package reference")
        
        # Check Procfile uses pixi run
        procfile_path = self.recipe_dir / "Procfile"
        with open(procfile_path) as f:
            procfile_content = f.read()
            if "pixi run" not in procfile_content:
                print("❌ Procfile missing 'pixi run' commands")
                return False
            print("✅ Procfile uses 'pixi run' commands")
        
        return True
    
    def test_app_configuration(self):
        """Test app.py configuration."""
        print("Testing app.py configuration...")
        
        app_path = self.recipe_dir / "app.py"
        with open(app_path) as f:
            content = f.read()
            
            # Check for essential imports
            required_imports = [
                "import gradio",
                "from colpali_engine",
                "from qdrant_client",
                "from rerankers"
            ]
            
            for import_line in required_imports:
                if import_line not in content:
                    print(f"❌ Missing import: {import_line}")
                    return False
            
            print("✅ App has all required imports")
            
            # Check for MAX references (should be updated from MAX Serve)
            if "MAX Serve" in content:
                print("❌ Found outdated 'MAX Serve' references")
                return False
            print("✅ No outdated 'MAX Serve' references found")
        
        return True
    
    def run_integration_test(self):
        """Run the full integration test."""
        print("=" * 70)
        print("MULTIMODAL-RAG-WITH-COLPALI-LLAMAVISION-RERANKER INTEGRATION TEST")
        print("=" * 70)
        
        try:
            # Test basic configuration
            if not self.test_basic_configuration():
                print("❌ Basic configuration test failed")
                return False
            
            # Test app configuration
            if not self.test_app_configuration():
                print("❌ App configuration test failed")
                return False
            
            # Check Docker availability
            if not self.check_docker_available():
                print("❌ Docker not available - skipping integration test")
                print("✅ Configuration tests passed!")
                return True
            
            # Check CUDA availability
            if not self.check_cuda_available():
                print("❌ No CUDA accelerators available - skipping integration test")
                print("✅ Configuration tests passed!")
                return True
            
            # Start services
            if not self.start_services():
                print("❌ Failed to start services")
                return False
            
            # Wait for services to be ready
            if not self.wait_for_services():
                print("❌ Services failed to start properly")
                return False
            
            print("✅ Integration test passed!")
            return True
            
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            return False
        finally:
            self.stop_services()


def run_quick_config_test():
    """Run a quick configuration test without starting services."""
    print("Running quick configuration test...")
    print("Note: This recipe requires NVIDIA GPU with 35GB+ VRAM (CUDA) - skipping dependency tests in CI")
    
    recipe_dir = Path(__file__).parent
    
    # Test that essential files exist
    essential_files = ["pyproject.toml", "metadata.yaml", "app.py", "Procfile"]
    for file_name in essential_files:
        file_path = recipe_dir / file_name
        if not file_path.exists():
            print(f"❌ Missing {file_name}")
            return False
        print(f"✅ {file_name} exists")
    
    # Test pyproject.toml has pixi configuration
    pyproject_path = recipe_dir / "pyproject.toml"
    with open(pyproject_path) as f:
        content = f.read()
        if "[tool.pixi" not in content:
            print("❌ pyproject.toml missing pixi configuration")
            return False
        print("✅ pyproject.toml has pixi configuration")
    
    # Test that it references pixi, not magic
    if "magic" in content.lower():
        print("❌ Found 'magic' references in pyproject.toml")
        return False
    print("✅ No 'magic' references found")
    
    # Test that it uses modular package
    if "modular" not in content:
        print("❌ Missing 'modular' package reference")
        return False
    print("✅ Found 'modular' package reference")
    
    # Check for CUDA system requirements
    if "[system-requirements]" not in content or "cuda" not in content:
        print("❌ Missing CUDA system requirements")
        return False
    print("✅ CUDA system requirements specified")
    
    # Check for CUDA-dependent packages
    if "torch" not in content or "torchvision" not in content:
        print("❌ Missing CUDA-dependent packages (torch/torchvision)")
        return False
    print("✅ CUDA-dependent packages present (torch/torchvision)")
    
    # Check Procfile uses pixi run
    procfile_path = recipe_dir / "Procfile"
    with open(procfile_path) as f:
        procfile_content = f.read()
        if "pixi run" not in procfile_content:
            print("❌ Procfile missing 'pixi run' commands")
            return False
        print("✅ Procfile uses 'pixi run' commands")
    
    # Check app.py doesn't have outdated references
    app_path = recipe_dir / "app.py"
    with open(app_path) as f:
        app_content = f.read()
        if "MAX Serve" in app_content:
            print("❌ Found outdated 'MAX Serve' references in app.py")
            return False
        print("✅ No outdated 'MAX Serve' references in app.py")
    
    print("✅ Quick configuration test passed!")
    print("Note: Full integration test requires NVIDIA GPU (CUDA) environment")
    return True


if __name__ == "__main__":
    # For CI environments, run quick test only
    if os.environ.get("CI") or "--quick" in sys.argv:
        print("Running in CI mode - quick configuration test only")
        success = run_quick_config_test()
    else:
        # Full integration test
        test = MultimodalRAGIntegrationTest()
        success = test.run_integration_test()
    
    if not success:
        sys.exit(1)
    
    print("🎉 All tests passed!")