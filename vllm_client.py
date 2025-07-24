import requests
import json
import re
# import msgspec
from vllm import SamplingParams
from typing import Optional, Dict, Any
from pydantic import ValidationError

class VLLMClient:
    # add json schema and use parameters to make it greedy
    def __init__(self, schema=None, url: str ="http://localhost:8000/generate") -> None:
        """
        Initialize the VLLM API wrapper.
        
        Args:
            url: Base URL of the VLLM server
        """
        self.url = url
        self.schema = schema
    
    def _convert_schema_to_json_example(self, prompt: str) -> str:
        """
        Convert Python class schemas in prompts to JSON examples.
        This prevents the LLM from returning Python code instead of JSON.
        """
        # Look for the schema section more broadly
        if 'You MUST adhere to this schema:' in prompt and 'class ' in prompt and 'BaseModel' in prompt:
            # Generate a JSON example based on the schema type
            if self.schema:
                try:
                    # Create a simple example instance and convert to JSON
                    if hasattr(self.schema, 'model_json_schema'):
                        schema_dict = self.schema.model_json_schema()
                        json_example = self._generate_json_example_from_schema(schema_dict)
                        
                        # Replace the schema instruction with JSON example instruction
                        new_prompt = prompt.replace(
                            'You MUST adhere to this schema:',
                            f'You MUST respond with valid JSON in this exact format:\n{json_example}\n\nDo not include any Python code or class definitions in your response.'
                        )
                        
                        # Remove the Python class definitions
                        # Find the start of the class definitions
                        schema_start = new_prompt.find('class ')
                        if schema_start != -1:
                            # Find the end by looking for the next major section or end of system message
                            schema_end = new_prompt.find('<|im_end|>', schema_start)
                            if schema_end == -1:
                                schema_end = new_prompt.find('\n\n', schema_start + 50)  # Look for double newline
                            if schema_end == -1:
                                schema_end = len(new_prompt)
                            
                            # Remove the class definitions
                            new_prompt = new_prompt[:schema_start] + new_prompt[schema_end:]
                        
                        return new_prompt
                except Exception as e:
                    print(f"Error in schema conversion: {e}")
        
        return prompt
    
    def _generate_json_example_from_schema(self, schema_dict: Dict[str, Any]) -> str:
        """Generate a JSON example from a Pydantic schema dictionary."""
        properties = schema_dict.get('properties', {})
        example = {}
        
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get('type', 'string')
            
            if prop_type == 'string':
                if 'description' in prop_info:
                    desc = prop_info['description']
                    if 'step' in prop_name.lower():
                        example[prop_name] = "Your reasoning step here"
                    elif 'strategy' in prop_name.lower():
                        example[prop_name] = "Your strategy here"
                    elif 'answer' in prop_name.lower():
                        example[prop_name] = "Your answer here"
                    else:
                        example[prop_name] = f"Your {prop_name} here"
                else:
                    example[prop_name] = f"example_{prop_name}"
            elif prop_type == 'number':
                example[prop_name] = 0.85
            elif prop_type == 'integer':
                example[prop_name] = 1
            elif prop_type == 'boolean':
                example[prop_name] = True
            elif prop_type == 'array':
                items = prop_info.get('items', {})
                if items.get('type') == 'string':
                    example[prop_name] = ["item1", "item2"]
                elif '$ref' in items or 'properties' in items:
                    # Handle nested objects in arrays
                    if prop_name == 'reasoning_steps' or 'step' in prop_name.lower():
                        example[prop_name] = [
                            {"step": "First reasoning step", "required_info": ["info1", "info2"]},
                            {"step": "Second reasoning step", "required_info": ["info3", "info4"]}
                        ]
                    else:
                        example[prop_name] = [{"example": "nested object"}]
                else:
                    example[prop_name] = ["example"]
        
        return json.dumps(example, indent=2)
    
    def generate(self, prompt: str, sampling_params: Optional[Dict[str, Any]]=None):
        """
        Generate text using VLLM with the specified sampling parameters.
        
        Args:
            prompt: Input prompt text
            sampling_params: sampling parameters as specified in the VLLM Sampling Params Object: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
        
        Returns:
            Response from the VLLM server as a dictionary
            
        Raises:
            Exception: If the API request fails
        """
        # Convert Python schemas to JSON examples automatically
        processed_prompt = self._convert_schema_to_json_example(prompt)
        
        payload: Dict[str, Any]  = {
            "prompt": processed_prompt
        }

        # breaks down samping params and adds it to the api call.
        if sampling_params:
            for key, value in sampling_params.items():
                payload[key] = value
        
        # Don't send json_schema to avoid 500 errors - parse response manually instead
        # if self.schema:
        #     payload["json_schema"] = self.schema.model_json_schema()
        
        response = requests.post(self.url, json=payload)
        try:
            # A common error is if the LLM loops through tokens until it hits the max limit, failing to output proper tokens for the schema.
            # Catch that error and regenerate the response using a repetition penalty to prevent looping
            unwrapped = self._unwrap(response)
        except (ValueError, requests.HTTPError, ValidationError) as e:
            print(f"Error in parsing LLM response: {e}")
            print(f"Prompt: {prompt[:200]}...")
            payload["repetition_penalty"] = 1.1
            response = requests.post(self.url, json=payload)
            unwrapped = self._unwrap(response)

        return unwrapped

    def _unwrap(self, response):
        # takes out the prompt from the response, and converts it to the expected schema, if any
        """Validate and parse a response from the VLLM server."""

        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from e
        
        if not isinstance(data, dict) or "text" not in data:
            raise ValueError("Invalid response format")
        
        # The VLLM server returns the full text including the prompt
        # We need to extract just the completion part after the final assistant tag
        full_text = data["text"][0]
        
        if not self.schema:
            # For non-schema responses, try to extract text after the last assistant tag
            if "<|im_start|>assistant" in full_text:
                assistant_parts = full_text.split("<|im_start|>assistant")
                if len(assistant_parts) > 1:
                    response_text = assistant_parts[-1].strip()
                    # Remove any trailing tags
                    if "<|im_end|>" in response_text:
                        response_text = response_text.split("<|im_end|>")[0].strip()
                    return response_text
            return full_text
            
        # For schema responses, extract and parse JSON
        # Look for JSON content after the assistant tag
        if "<|im_start|>assistant" in full_text:
            assistant_parts = full_text.split("<|im_start|>assistant")
            if len(assistant_parts) > 1:
                response_text = assistant_parts[-1].strip()
                # Remove any trailing tags
                if "<|im_end|>" in response_text:
                    response_text = response_text.split("<|im_end|>")[0].strip()
            else:
                response_text = full_text
        else:
            response_text = full_text
        
        # Clean up the response text - remove markdown code blocks and extract JSON
        cleaned_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if cleaned_text.startswith('```'):
            lines = cleaned_text.split('\n')
            # Remove first line (```language) and last line (```)
            if lines[-1].strip() == '```':
                cleaned_text = '\n'.join(lines[1:-1])
            else:
                # Find the closing ``` and remove everything after it
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == '```':
                        cleaned_text = '\n'.join(lines[1:i])
                        break
        
        # Try to find JSON content in the response
        json_start = cleaned_text.find('{')
        json_end = cleaned_text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_content = cleaned_text[json_start:json_end+1]
            try:
                # Try to parse as JSON first
                parsed = json.loads(json_content)
                return self.schema.parse_obj(parsed)
            except (json.JSONDecodeError, ValueError):
                pass
        
        # If JSON parsing failed, try the original text
        try:
            return self.schema.parse_raw(cleaned_text)
        except ValueError:
            # Last resort: try to parse the original text as JSON
            try:
                parsed = json.loads(cleaned_text)
                return self.schema.parse_obj(parsed)
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse response as valid JSON for schema {self.schema.__name__}. Response: {response_text[:200]}...")

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

