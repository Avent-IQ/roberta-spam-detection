# Roberta Base Quantized Model for Spam Detection

This repository hosts a quantized version of the **roberta-base** model, fine-tuned for **spam detection** tasks. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## Model Details

- **Model Architecture:** Roberta Base  
- **Task:** Spam Detection  
- **Dataset:** Hugging Face's `sms_spam`, `spam_mail`, and `mail_spam_ham_dataset`  
- **Quantization:** Float16  
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage

### Installation

```sh
pip install transformers torch
```

### Loading the Model

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "AventIQ-AI/roberta-spam-detection"
model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = RobertaTokenizer.from_pretrained(model_name)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
    
    return "Spam" if predicted_class == 1 else "Ham"

# Sample test messages
input_text = "Congratulations! You have won a free iPhone. Click here to claim your prize."
print(f"Prediction: {predict(input_text)}")  # Expected output: Spam
```

## 📊 Classification Report (Quantized Model - bfloat16)
 
| Metric      | Class 0 (Non-Spam) | Class 1 (Spam) | Macro Avg | Weighted Avg |
|------------|----------------|----------------|------------|--------------|
| **Precision** | 1.00           | 0.98           | 0.99       | 0.99         |
| **Recall**    | 0.99           | 0.99           | 0.99       | 0.99         |
| **F1-Score**  | 0.99           | 0.99           | 0.99       | 0.99         |
| **Accuracy**  | **99%**        | **99%**        | **99%**    | **99%**      |
 
### 🔍 **Observations**
✅ **Precision:** High (1.00 for non-spam, 0.98 for spam) → **Few false positives**  
✅ **Recall:** High (0.99 for both classes) → **Few false negatives**  
✅ **F1-Score:** **Near-perfect balance** between precision & recall  

## Fine-Tuning Details

### Dataset

The Hugging Face's `sms_spam`, `spam_mail`, and `mail_spam_ham_dataset` dataset was used, containing both spam and ham (non-spam) examples.

### Training

- Number of epochs: 3 
- Batch size: 8  
- Evaluation strategy: epoch  
- Learning rate: 3e-5

### Quantization

Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Fine Tuned Model
├── README.md            # Model documentation
```

## Limitations

- The model may not generalize well to domains outside the fine-tuning dataset.  
- Quantization may result in minor accuracy degradation compared to full-precision models.  

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

