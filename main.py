from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CLASES ────────────────────────────────────────────────
clases_10 = [
    'Acne/Rosacea', 'Actinic Keratosis/BCC', 'Eczema',
    'Pigmentation Disorders', 'Nail Fungus', 'Psoriasis/Lichen Planus',
    'Seborrheic Keratosis', 'Systemic Disease', 'Tinea/Candidiasis',
    'Warts/Molluscum'
]

clases_23 = [
    'Acne/Rosacea', 'Actinic Keratosis/BCC', 'Atopic Dermatitis',
    'Bullous Disease', 'Cellulitis/Impetigo', 'Eczema',
    'Drug Eruptions', 'Alopecia', 'Herpes/HPV',
    'Pigmentation Disorders', 'Lupus', 'Melanoma/Nevi',
    'Nail Fungus', 'Contact Dermatitis', 'Psoriasis/Lichen Planus',
    'Scabies/Lyme', 'Seborrheic Keratosis', 'Systemic Disease',
    'Tinea/Candidiasis', 'Urticaria', 'Vascular Tumors',
    'Vasculitis', 'Warts/Molluscum'
]

descripciones = {
    'Acne/Rosacea': 'Afección inflamatoria del folículo piloso que produce granos, pústulas y enrojecimiento facial.',
    'Actinic Keratosis/BCC': 'Lesiones escamosas precancerosas causadas por exposición solar. El carcinoma basocelular es el cáncer de piel más frecuente.',
    'Atopic Dermatitis': 'Enfermedad inflamatoria crónica con picor intenso y piel seca, frecuente en pliegues.',
    'Bullous Disease': 'Enfermedades autoinmunes que producen ampollas en la piel y mucosas.',
    'Cellulitis/Impetigo': 'Infecciones bacterianas de la piel que producen enrojecimiento, calor e hinchazón.',
    'Eczema': 'Inflamación crónica con picor, enrojecimiento y descamación de la piel.',
    'Drug Eruptions': 'Reacciones cutáneas causadas por medicamentos.',
    'Alopecia': 'Pérdida de cabello parcial o total que puede seguir distintos patrones.',
    'Herpes/HPV': 'Infecciones víricas que producen vesículas o verrugas en la piel.',
    'Pigmentation Disorders': 'Alteraciones en la distribución de la melanina que producen manchas claras u oscuras.',
    'Lupus': 'Enfermedad autoinmune que produce eritema facial en forma de mariposa.',
    'Melanoma/Nevi': 'El melanoma es el cáncer de piel más agresivo. Los nevos son lunares que pueden evolucionar.',
    'Nail Fungus': 'Infección fúngica de las uñas con engrosamiento y decoloración.',
    'Contact Dermatitis': 'Reacción inflamatoria por contacto con alérgenos o irritantes.',
    'Psoriasis/Lichen Planus': 'Enfermedades crónicas que producen placas escamosas o pápulas violáceas.',
    'Scabies/Lyme': 'La sarna produce surcos y picor intenso. La enfermedad de Lyme produce eritema en diana.',
    'Seborrheic Keratosis': 'Tumores benignos de aspecto verrugoso frecuentes en personas mayores.',
    'Systemic Disease': 'Manifestaciones cutáneas de enfermedades internas como diabetes o enfermedades tiroideas.',
    'Tinea/Candidiasis': 'Infecciones fúngicas superficiales que producen lesiones circulares o en pliegues.',
    'Urticaria': 'Reacción alérgica con habones rojizos y pruriginosos que aparecen y desaparecen.',
    'Vascular Tumors': 'Tumores originados en vasos sanguíneos, como hemangiomas.',
    'Vasculitis': 'Inflamación de vasos sanguíneos que produce púrpura o úlceras en la piel.',
    'Warts/Molluscum': 'Lesiones víricas con superficie rugosa o pápulas redondeadas con depresión central.',
}

# ── TRANSFORM ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── CARGAR MODELOS ────────────────────────────────────────
def cargar_modelo(ruta, num_clases):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_clases)
    )
    model.load_state_dict(torch.load(ruta, map_location='cpu'))
    model.eval()
    return model

MODEL_10 = cargar_modelo(
    r"C:\Users\jiaxi\OneDrive\Documentos\A_TFG\efficientnetb0_10clases.pth", 10)
MODEL_23 = cargar_modelo(
    r"C:\Users\jiaxi\OneDrive\Documentos\A_TFG\efficientnetb0_23clases.pth", 23)

print("Modelos cargados correctamente.")

# ── PREDICCIÓN ────────────────────────────────────────────
def predecir(model, clases, imagen_bytes):
    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)
    top3 = torch.topk(probs, 3)
    resultados = []
    for i in range(3):
        idx = top3.indices[i].item()
        prob = top3.values[i].item()
        nombre = clases[idx]
        resultados.append({
            "clase": nombre,
            "probabilidad": round(prob * 100, 2),
            "descripcion": descripciones.get(nombre, "")
        })
    return resultados

# ── ENDPOINTS ─────────────────────────────────────────────
@app.post("/predict/10clases")
async def predict_10(file: UploadFile = File(...)):
    contenido = await file.read()
    return {"modelo": "10 clases", "resultados": predecir(MODEL_10, clases_10, contenido)}

@app.post("/predict/23clases")
async def predict_23(file: UploadFile = File(...)):
    contenido = await file.read()
    return {"modelo": "23 clases", "resultados": predecir(MODEL_23, clases_23, contenido)}

# ── SERVIR FRONTEND ───────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")