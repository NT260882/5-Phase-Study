import json
import ollama
from pydantic import BaseModel, Field, ValidationError

# Define the strict target Pydantic schema
class DeveloperProfile(BaseModel):
    name: str = Field(description="Developer's full name")
    primary_language: str = Field(description="Main programming language")
    years_experience: int = Field(description="Years of professional coding experience")
    skills: list[str] = Field(description="List of core technical skills")

SYSTEM_SCHEMA_PROMPT = f"""You are a strict JSON generation engine.
You MUST respond with a valid JSON object matching this schema EXACTLY:
{json.dumps(DeveloperProfile.model_json_schema(), indent=2)}

Do not include markdown blocks, intro text, or conversational commentary. Output ONLY raw JSON."""

def generate_structured_profile(user_input: str, model_name: str = "llama3.2", temperature: float = 0.0, max_retries: int = 2):
    """
    Forces structured JSON output, validates it against a Pydantic schema,
    and runs a self-correction retry loop if parsing fails.
    """
    current_prompt = f"Extract a developer profile from this input text:\n'{user_input}'"
    
    for attempt in range(max_retries + 1):
        print(f"\n⚡ Ingestion Attempt {attempt + 1} (Temp: {temperature})")
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_SCHEMA_PROMPT},
                {"role": "user", "content": current_prompt}
            ],
            options={"temperature": temperature}
        )
        
        raw_output = response["message"]["content"].strip()
        
        # Clean potential markdown formatting
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        try:
            # 1. Validate raw JSON
            parsed_json = json.loads(raw_output)
            # 2. Validate against Pydantic schema
            validated_data = DeveloperProfile(**parsed_json)
            print("✅ Successfully Validated Output Payload:")
            print(validated_data.model_dump_json(indent=2))
            return validated_data

        except (json.JSONDecodeError, ValidationError) as err:
            print(f"❌ Validation Failed on Attempt {attempt + 1}: {err}")
            
            if attempt < max_retries:
                # Construct the self-correction re-prompt payload
                current_prompt = f"""Your previous output failed validation with error: {err}
Raw Output was: {raw_output}

Please fix the error and return the corrected JSON matching the schema strictly."""
                print("🔄 Executing Self-Correction Re-prompt Loop...")
            else:
                print("❌ Exceeded maximum retries. Returning None.")
                return None

if __name__ == "__main__":
    sample_text = "Naveen is a principal engineer with 8 years of experience building Python and Golang systems."
    
    print("\n================ TEST 1: LOW TEMPERATURE (0.0) ================")
    generate_structured_profile(sample_text, temperature=0.0)

    print("\n================ TEST 2: HIGH TEMPERATURE (0.8) ================")
    generate_structured_profile(sample_text, temperature=0.8)