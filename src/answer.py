"""Build an answer from retrieved articles. Uses an LLM if an API key is set
(GROQ_API_KEY, OpenAI-compatible), otherwise returns an extractive answer with citations."""
import os

import requests

SYSTEM = ("أنت مساعد قانوني يجيب بالعربية المبسطة عن أسئلة قانون العمل المصري. "
          "اعتمد فقط على المواد المرفقة، واذكر رقم المادة بين قوسين بعد كل معلومة مثل (مادة 47). "
          "إذا لم تجد الإجابة في المواد المرفقة فقل ذلك بوضوح ولا تخمّن. "
          "اختم بتنبيه أن هذه معلومات عامة وليست استشارة قانونية.")


def format_context(arts):
    return "\n\n".join(f"[{a['law']} - {a['id']}]\n{a['text']}" for a in arts)


def llm_answer(question, arts, model=None):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": 0.1, "messages": [
                          {"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"المواد:\n{format_context(arts)}\n\nالسؤال: {question}"}]},
                      timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extractive_answer(arts):
    lines = ["أقرب المواد لسؤالك:"]
    for a in arts[:3]:
        snippet = a["text"][:450] + ("…" if len(a["text"]) > 450 else "")
        lines.append(f"\n**{a['id']}** — {a.get('chapter') or a.get('part') or ''}\n{snippet}")
    lines.append("\n_معلومات عامة وليست استشارة قانونية._")
    return "\n".join(lines)


def answer(question, retriever, k=5):
    arts = retriever.search(question, k=k)
    try:
        text = llm_answer(question, arts)
    except Exception as e:
        text = None
        print(f"[answer] LLM failed, falling back to extractive: {e}")
    return (text or extractive_answer(arts)), arts
