"""Gradio demo: ask questions about the Egyptian Labour Law in Arabic."""
import gradio as gr

from src.answer import answer
from src.retriever import HybridRetriever, load_articles

retriever = HybridRetriever(load_articles())


def chat(message, history):
    text, arts = answer(message, retriever)
    sources = "\n".join(f"- {a['id']} ({a['law']})" for a in arts)
    return f"{text}\n\n**المصادر:**\n{sources}"


demo = gr.ChatInterface(
    chat,
    title="مساعد قانون العمل المصري",
    description="اسأل بالعربي أو بالمصري عن قانون العمل رقم 14 لسنة 2025. الإجابات مبنية على نص المواد مع ذكر رقم كل مادة.",
    examples=["كام يوم أجازة سنوية من حقي؟", "صاحب العمل يقدر يفصلني من غير إنذار؟", "ساعات العمل القانونية كام في اليوم؟"],
    rtl=True,
)

if __name__ == "__main__":
    demo.launch()
