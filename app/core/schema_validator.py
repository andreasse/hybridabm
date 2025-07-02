"""Generic JSON Schema validator for clean architecture."""
import json
from typing import Dict, Any, List
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError


class SchemaValidator:
    """Generic validator that loads JSON Schema dynamically."""
    
    def __init__(self, schema_path: str):
        """Initialize with path to JSON Schema file."""
        self.schema_path = Path(schema_path)
        self._schema = None
    
    @property
    def schema(self) -> Dict[str, Any]:
        """Dynamically load and cache the JSON Schema."""
        if self._schema is None:
            if self.schema_path.exists():
                with open(self.schema_path, 'r') as f:
                    self._schema = json.load(f)
            else:
                raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        return self._schema
    
    def validate(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate data against the JSON Schema.
        
        Args:
            data: Data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            validate(instance=data, schema=self.schema)
        except ValidationError as e:
            # Format jsonschema errors in a user-friendly way
            path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            errors.append(f"Validation error at {path}: {e.message}")
        except Exception as e:
            errors.append(f"Schema validation failed: {str(e)}")
        
        return errors
    
    def reload_schema(self):
        """Force reload of schema (useful for development)."""
        self._schema = None


# Global validator instance for simulation parameters
simulation_validator = SchemaValidator("/home/andreas/hybrid_article/simulation-params-contract-final.json")