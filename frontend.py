import gradio as gr
from config import get_logger
from datetime import datetime

logger = get_logger(__name__)

def create_interface(
    load_context_wrapper,
    answer_question_wrapper,
    process_faq_wrapper,
    set_extraction_method
):
    logger.info("Criando interface Gradio")

    chat_history = []

    with gr.Blocks(title="Processo Jurídico - Extrator Autos") as iface:
        gr.Markdown("# Processo Jurídico - Extrator Autos")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload de PDF")
                extraction_method = gr.Radio(
                    ["ocrmypdf", "pdf2image"],
                    label="Método de Extração",
                    value="ocrmypdf"
                )
                load_context_button = gr.Button("Processar Documento")

        with gr.Row():
            context_status = gr.Textbox(label="Status do Processamento", interactive=False)

        with gr.Column(visible=False) as faq_elements:
            faq_button = gr.Button("Processar FAQ")
            faq_result = gr.Textbox(label="Resultado do FAQ", lines=10)

        with gr.Column(visible=False) as chat_elements:
            question_input = gr.Textbox(lines=2, placeholder="Faça a sua pergunta aqui...", label="Faça a sua Pergunta")
            submit_question = gr.Button("Enviar Pergunta")
            answer_output = gr.Textbox(label="Resposta", lines=10)

            chat_history_component = gr.Dataframe(
                headers=["Pergunta", "Resposta", "Tempo de Resposta (s)"],
                datatype=["str", "str", "number"],
                label="Histórico de Perguntas e Respostas",
                value=chat_history
            )

        def process_load_context(pdf, method):
            nonlocal chat_history
            chat_history = []
            set_extraction_method(method)

            try:
                for result in load_context_wrapper(pdf):
                    if not result["success"]:
                        yield (
                            result["status"],
                            gr.update(interactive=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            chat_history
                        )
                    else:
                        yield (
                            result["status"],
                            gr.update(interactive=True),
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(interactive=True),
                            gr.update(interactive=True),
                            chat_history
                        )
            except Exception as e:
                logger.error(f"Erro durante o processamento: {str(e)}")
                yield (
                    f"Erro durante o processamento: {str(e)}",
                    gr.update(interactive=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    chat_history
                )

        def process_question(question):
            nonlocal chat_history
            start_time = datetime.now()

            try:
                answer = answer_question_wrapper(question)
                end_time = datetime.now()
                time_taken = (end_time - start_time).total_seconds()

                chat_history.append([question, answer, round(time_taken, 2)])

                return answer, chat_history
            except Exception as e:
                logger.error(f"Erro ao processar pergunta: {str(e)}")
                return f"Erro ao processar pergunta: {str(e)}", chat_history

        def process_faq():
            try:
                faq_result = process_faq_wrapper()
                return gr.update(value=faq_result, visible=True), gr.update(interactive=False), gr.update(interactive=False)
            except Exception as e:
                logger.error(f"Erro ao processar FAQ: {str(e)}")
                return gr.update(value=f"Erro ao processar FAQ: {str(e)}", visible=True), gr.update(interactive=True), gr.update(interactive=True)

        def on_faq_completion(faq_result):
            return gr.update(interactive=True), gr.update(interactive=True)

        load_context_button.click(
            fn=process_load_context,
            inputs=[pdf_input, extraction_method],
            outputs=[context_status, load_context_button, faq_elements, chat_elements, faq_button, submit_question, chat_history_component]
        )

        submit_question.click(
            fn=process_question,
            inputs=[question_input],
            outputs=[answer_output, chat_history_component]
        )

        faq_button.click(
            fn=process_faq,
            inputs=[],
            outputs=[faq_result, submit_question, faq_button]
        ).then(
            fn=on_faq_completion,
            inputs=[],
            outputs=[submit_question, faq_button]
        )

    return iface

def launch_interface(iface):
    logger.info("Lançando app Gradio")
    iface.launch(server_name="0.0.0.0", debug=True, server_port=7863)