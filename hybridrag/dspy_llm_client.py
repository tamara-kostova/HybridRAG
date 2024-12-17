import dspy
import requests
import os


class LlamaDSPyModel(dspy.LM):
    def __init__(self, host=None, port=11434, model="llama3:8b", max_tokens=300):
        super().__init__(model)
        self.host = host or os.getenv("LLM_IP_ADDRESS", "localhost")
        self.port = port
        self.model = model
        self.max_tokens = max_tokens

    def _generate(self, prompt):
        try:
            response = requests.post(
                url=f"http://{self.host}:{self.port}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def __call__(self, prompt, **kwargs):
        generation = self._generate(prompt)
        return dspy.Prediction(text=generation, **kwargs)


def configure_dspy_llama():
    llm = LlamaDSPyModel()
    dspy.settings.configure(lm=llm)
    return llm
