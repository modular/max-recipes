from typing import Any

import reflex as rx
from reflex.event import EventSpec

# from .agent import process_query


class State(rx.State):
    query: str | None
    response: str | None

    async def handle_submit(self, form_data: dict[str, Any]) -> EventSpec | None:
        try:
            if isinstance(form_data["query"], str):
                self.query = form_data["query"]
                # res = await process_query(self.query)
                # resp = f"There character {res.character_found}"
                # resp += " appears in the string {res.in_string} {res.num_times} "
                # resp += "time" if res.num_times == 1 else "times"
                # self.response = resp
            else:
                raise ValueError("Form data did not include a query string")
        except Exception as e:
            return rx.console_log(f"Error handling query:\n{e}")


def form() -> rx.Component:
    return rx.form.root(
        rx.hstack(
            rx.form.field(
                rx.form.control(
                    as_child=True,
                ),
                name="query",
            ),
            rx.form.submit(rx.button("Send"), as_child=True),
        ),
        reset_on_submit=True,
        on_submit=State.handle_submit,  # type: ignore
    )


def index() -> rx.Component:
    return rx.vstack(
        form(),
        rx.text(State.query),
        rx.text(State.response),
    )


app = rx.App()
app.add_page(index)
