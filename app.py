import os, torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.getenv("MODEL_ID", "bnaz12/kambaata-gpt-mini")
HF_TOKEN = os.getenv("HF_TOKEN")          # only needed while the model repo is private

torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN).eval()
MAX_CTX = model.config.n_positions        # 256

app = FastAPI(title="Kambaata text generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenRequest(BaseModel):
    prompt: str = Field("", max_length=300)
    max_new_tokens: int = Field(60, ge=10, le=200)      # about 1 word per 1.6 tokens
    temperature: float = Field(0.8, ge=0.1, le=1.5)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/api/generate")
def generate(req: GenRequest):
    ids = [tok.bos_token_id] + tok.encode(req.prompt.strip(), add_special_tokens=False)
    ids = ids[-(MAX_CTX - req.max_new_tokens):]          # keep prompt + output within the context window
    x = torch.tensor([ids])
    with torch.no_grad():
        out = model.generate(
            x, attention_mask=torch.ones_like(x),
            max_new_tokens=req.max_new_tokens, do_sample=True,
            temperature=req.temperature, top_k=50, repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id, eos_token_id=None,     # keep going past </s>
        )
    text = tok.decode(out[0], skip_special_tokens=False)
    text = text.replace("<pad>", "").replace("<s>", "").replace("</s>", "\n")
    text = "\n".join(l.strip() for l in text.split("\n") if l.strip())
    return {"text": text, "new_tokens": int(out.shape[1] - len(ids))}

@app.get("/")
def home():
    return FileResponse("index.html")
