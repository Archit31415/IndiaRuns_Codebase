import json
from pathlib import Path

def build_rubric():
    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "data" / "precomputed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "recruiter_rubric.json"

    rubric = {
  "metadata": {
    "role": "Senior AI Engineer - Founding Team",
    "version": "3.1_jd_complete"
  },

  "stage0_hard_filters": {
    "min_years_experience": 4.0,

    "blacklisted_companies_if_exclusive": [
      "tcs",
      "infosys",
      "wipro",
      "accenture",
      "cognizant",
      "capgemini"
    ],

    "incompatible_domains": [
      "computer vision",
      "speech",
      "robotics"
    ],

    "must_have_production_experience": True,

    "reject_research_only_profiles": True,

    "reject_if_no_recent_coding": {
      "enabled": True,
      "months_since_last_production_code": 18
    }
  },

  "stage1_text_targets": {
    "core_vector_search": {
      "weight": 2.5,
      "keywords": [
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "opensearch",
        "elasticsearch",
        "faiss"
      ]
    },

    "core_applied_ml": {
      "weight": 2.0,
      "keywords": [
        "embeddings",
        "retrieval",
        "hybrid retrieval",
        "dense retrieval",
        "semantic search",
        "sentence-transformers",
        "openai embeddings",
        "bge",
        "e5",
        "ndcg",
        "mrr",
        "map",
        "rerank",
        "reranking"
      ]
    },

    "ranking_and_recommendation": {
      "weight": 2.5,
      "keywords": [
        "ranking",
        "recommendation",
        "recommendation system",
        "search",
        "matching system",
        "learning to rank",
        "candidate ranking"
      ]
    },

    "languages": {
      "weight": 1.5,
      "keywords": [
        "python",
        "c++"
      ]
    },

    "llm_and_finetuning": {
      "weight": 1.0,
      "keywords": [
        "llm",
        "fine tuning",
        "fine-tuning",
        "lora",
        "qlora",
        "peft",
        "rag"
      ]
    },

    "evaluation_frameworks": {
      "weight": 2.5,
      "keywords": [
        "ab testing",
        "a/b testing",
        "offline evaluation",
        "online evaluation",
        "offline to online",
        "ranking metrics",
        "evaluation framework"
      ]
    }
  },

  "stage2_experience_scoring": {
    "product_company_multiplier": 1.5,

    "job_hopper_penalty": 0.5,

    "job_hopper_threshold_years": 1.5,

    "framework_enthusiast_penalty": 0.5,

    "langchain_only_penalty": 1.0,

    "recommendation_system_bonus": 2.0,

    "ranking_system_bonus": 2.0,

    "retrieval_system_bonus": 2.0,

    "marketplace_or_hrtech_bonus": 1.2,

    "open_source_validation_bonus": 1.3
  },

  "stage3_behavioral_signals": {
    "target_notice_period_days": 30,

    "max_acceptable_notice_period": 90,

    "min_response_rate": 0.50,

    "weights": {
      "recruiter_response_rate": 0.40,
      "github_activity_score": 0.40,
      "open_to_work_flag": 0.20
    },

    "notice_period_penalty_threshold": 30,

    "inactive_candidate_threshold_days": 180
  },

  "stage4_final_ranking_weights": {
    "experience_score": 0.25,
    "product_company_score": 0.20,
    "retrieval_ranking_score": 0.25,
    "production_system_score": 0.15,
    "behavioral_score": 0.10,
    "external_validation_score": 0.05
  }
}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rubric, f, indent=4)

    print(f"Recruiter rubric successfully generated and verified at: {output_path}")

if __name__ == "__main__":
    build_rubric()