from transformers import pipeline

# Load the question-answering pipeline with a model fine-tuned on SQuAD
nlp = pipeline("question-answering")

# Example text and question
context = "The capital of France is Paris. It is located on the Seine River."
question = "What is the main topic of this paragraph?"

# Get the answer, which often encapsulates the key concept
answer = nlp(question=question, context=context)["answer"]
print("Answer:", answer)  # Output: Answer: The capital of France

# Further analysis to extract more specific concepts:

# Tokenize the context and question
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

#model_name = "bert-base-uncased"  # Choose a suitable model
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# Analyze the attention patterns to identify important tokens
attention_weights = outputs.attentions[-1]  # Last layer attention

# Extract highly attended tokens and analyze their semantic roles
# for potential concept identification using additional NLP techniques