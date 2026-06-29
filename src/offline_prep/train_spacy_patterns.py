import json
import spacy
from pathlib import Path

def train_spacy_patterns():
    """
    Builds a lightning-fast, rule-based spaCy NER pipeline by dynamically 
    parsing the 'stage1_text_targets' from the recruiter rubric.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    precomputed_dir = base_dir / "data" / "precomputed"
    rubric_path = precomputed_dir / "recruiter_rubric.json"
    model_output_dir = precomputed_dir / "local_models" / "spacy_ner_pipeline"

    if not rubric_path.exists():
        print("Error: Rubric not found.")
        return

    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = json.load(f)

    print("Initializing blank English spaCy pipeline...")
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    
    patterns = []
    text_targets = rubric.get("stage1_text_targets", {})
    
    for category, config in text_targets.items():
        label = category.upper()
        
        for keyword in config.get("keywords", []):
            words = keyword.split()
            if len(words) > 1:
                pattern = [{"LOWER": w.lower()} for w in words]
            else:
                pattern = [{"LOWER": keyword.lower()}]
                
            patterns.append({"label": label, "pattern": pattern})

    print(f"Injecting {len(patterns)} entity patterns mapped from the rubric...")
    ruler.add_patterns(patterns)

    model_output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(model_output_dir)

    print(f"Custom spaCy NER pipeline compiled and saved to: {model_output_dir}")

if __name__ == "__main__":
    train_spacy_patterns()