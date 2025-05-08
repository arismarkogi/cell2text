from transformers import pipeline

# Initialize the model pipeline
generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",  # replace with actual local path
    device=-1  # set to -1 for CPU, 0 for first GPU
)

# Example structured entry
text_desc_list = [
    "Assay: 10x 3’ v3.; Cell type: blood vessel endothelial cell, an endothelial cell that lines the vasculature.; Development stage: 24-year-old human stage, a young adult stage referring to an adult over 24 and under 25 years old.; Disease: normal, indicating no deviation from normal or average.; Sex: female, referring to a biological sex quality in individuals that produce gametes that can be fertilized by male gametes.; Tissue: breast, the upper ventral region of the torso of an organism.",
    # Add more entries here
]




def make_prompt_v2(desc):
    return f"""You are a scientific assistant. Your task is to convert structured biological metadata into a concise scientific annotation of no more than 100 words.

Example:
Structured metadata:
Assay: 10x 3’ v2.; Cell type: cortical neuron, a neuron located in the brain cortex.; Development stage: fetal stage.; Disease: normal.; Sex: female.; Tissue: brain cortex.
Annotation:
A cortical neuron sample was collected from the brain cortex of a healthy female fetus and profiled using the 10x 3’ v2 assay. No disease was present.

End of example.

Now convert the following:

Structured metadata:
{desc}
Annotation:"""





# Run generation
for i, desc in enumerate(text_desc_list):
    prompt = make_prompt_v2(desc)
    result = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    generated_text = result[0]['generated_text']
    
    # Extract only the annotation part
    annotation = generated_text.split("Annotation:")[-1].strip()
    
    print(f"Sample {i+1} Annotation:\n{annotation}\n{'-'*60}")
