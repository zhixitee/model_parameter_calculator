import os
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import AutoModelForMaskGeneration, AutoModelForKeypointDetection, AutoModelForTextEncoding, AutoModelForImageToImage, AutoModel, AutoModelForPreTraining, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForTableQuestionAnswering, AutoModelForVisualQuestionAnswering, AutoModelForDocumentQuestionAnswering, AutoModelForTokenClassification, AutoModelForMultipleChoice, AutoModelForNextSentencePrediction, AutoModelForImageClassification, AutoModelForZeroShotImageClassification, AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation, AutoModelForInstanceSegmentation, AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection, AutoModelForDepthEstimation, AutoModelForVideoClassification, AutoModelForVision2Seq, AutoModelForAudioClassification, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoModelForAudioFrameClassification, AutoModelForAudioXVector, AutoModelForTextToSpectrogram, AutoModelForTextToWaveform, AutoBackbone, AutoModelForMaskedImageModeling
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API token from the environment
hf_api_token = os.getenv("HF_API_TOKEN")

def count_parameters(model_name):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=hf_api_token)
    
    model_classes = [
        AutoModelForMaskGeneration,
        AutoModelForKeypointDetection, 
        AutoModelForTextEncoding, 
        AutoModelForImageToImage, 
        AutoModel, 
        AutoModelForPreTraining, 
        AutoModelForCausalLM, 
        AutoModelForMaskedLM, 
        AutoModelForSeq2SeqLM, 
        AutoModelForSequenceClassification, 
        AutoModelForQuestionAnswering, 
        AutoModelForTableQuestionAnswering, 
        AutoModelForVisualQuestionAnswering, 
        AutoModelForDocumentQuestionAnswering, 
        AutoModelForTokenClassification, 
        AutoModelForMultipleChoice, 
        AutoModelForNextSentencePrediction, 
        AutoModelForImageClassification, 
        AutoModelForZeroShotImageClassification, 
        AutoModelForImageSegmentation, 
        AutoModelForSemanticSegmentation, 
        AutoModelForUniversalSegmentation, 
        AutoModelForInstanceSegmentation, 
        AutoModelForObjectDetection, 
        AutoModelForZeroShotObjectDetection, 
        AutoModelForDepthEstimation, 
        AutoModelForVideoClassification, 
        AutoModelForVision2Seq, 
        AutoModelForAudioClassification, 
        AutoModelForCTC, 
        AutoModelForSpeechSeq2Seq, 
        AutoModelForAudioFrameClassification, 
        AutoModelForAudioXVector, 
        AutoModelForTextToSpectrogram, 
        AutoModelForTextToWaveform, 
        AutoBackbone, 
        AutoModelForMaskedImageModeling
    ]
    
    model = None
    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(model_name, config=config, trust_remote_code=True, use_auth_token=hf_api_token)
            break
        except (ValueError, OSError):
            continue
    
    if model is None:
        raise ValueError(f"Could not load any supported model type for {model_name}. Please check the model repository.")
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_millions = total_params / 1e9
    model_size_gb = (total_params * 4) / (1024**3)
    
    return total_params, model_size_millions, model_size_gb

model_name = "lysandre/tiny-tapas-random-wtq"  # Replace with any model name from Hugging Face
total_params, model_size_millions, model_size_gb = count_parameters(model_name)

print(f"Total parameters: {total_params:,}")
print(f"Model size (in billions): {model_size_millions:.2f}B")
print(f"Approximate model size in GB: {model_size_gb:.2f} GB")
