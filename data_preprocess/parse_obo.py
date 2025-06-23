#!/usr/bin/env python3
"""
OBO to JSON Converter

A Python script to parse OBO (Open Biomedical Ontologies) files and convert them to JSON format.
Extracts terms with their names, definitions, and synonyms.

Usage:
    python obo_to_json.py <obo_file1> [obo_file2] ... -o <output.json>
    python obo_to_json.py cl.obo uberon-basic.obo -o ontologies.json
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class OBOTerm:
    """Represents a term in an OBO ontology."""
    id: str
    name: str = ""
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)
    is_obsolete: bool = False


class OBOToJSONConverter:
    """Converter for OBO format ontology files to JSON."""
    
    def __init__(self):
        self.terms: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            'total_terms': 0,
            'obsolete_terms': 0,
            'files_processed': 0
        }
    
    def parse_files(self, filepaths: List[str]) -> None:
        """
        Parse multiple OBO files and combine their terms.
        
        Args:
            filepaths: List of paths to OBO files
        """
        for filepath in filepaths:
            print(f"Processing {filepath}...")
            try:
                self.parse_file(filepath)
                self.stats['files_processed'] += 1
                print(f"✓ Successfully processed {filepath}")
            except Exception as e:
                print(f"✗ Error processing {filepath}: {e}")
    
    def parse_file(self, filepath: str) -> None:
        """
        Parse an OBO file and extract term information.
        
        Args:
            filepath: Path to the OBO file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into blocks
        blocks = self._split_into_blocks(content)
        
        # Parse each term block
        for block in blocks:
            if not block.strip():
                continue
                
            lines = [line.strip() for line in block.strip().split('\n')]
            if not lines:
                continue
                
            block_type = lines[0].strip('[]')
            
            if block_type == 'Term':
                term = self._parse_term_block(lines[1:])
                if term and term.id:
                    # Convert to the desired JSON format
                    term_data = {
                        "name": term.name,
                        "def": self._clean_definition(term.definition),
                    }
                    
                    # Only include synonym field if there are synonyms
                    if term.synonyms:
                        if len(term.synonyms) == 1:
                            term_data["synonym"] = term.synonyms[0]
                        else:
                            term_data["synonym"] = term.synonyms
                    
                    self.terms[term.id] = term_data
                    self.stats['total_terms'] += 1
                    
                    if term.is_obsolete:
                        self.stats['obsolete_terms'] += 1
    
    def _clean_definition(self, definition: str) -> str:
        """
        Clean up definition text by removing unwanted formatting.
        
        Args:
            definition: Raw definition string
            
        Returns:
            Cleaned definition string
        """
        if not definition:
            return ""
        
        # Replace literal \n with actual spaces
        cleaned = definition.replace('\\n', ' ')
        
        # Remove trailing backslashes that might indicate incomplete parsing
        cleaned = re.sub(r'\\+$', '', cleaned)
        
        # Normalize whitespace (replace multiple spaces/tabs with single space)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _split_into_blocks(self, content: str) -> List[str]:
        """Split OBO content into blocks based on [Term], [Typedef], etc."""
        # Find all block headers
        block_pattern = re.compile(r'^\[([^\]]+)\]', re.MULTILINE)
        matches = list(block_pattern.finditer(content))
        
        if not matches:
            return [content]
        
        blocks = []
        
        # Add each block
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            blocks.append(content[start:end])
        
        return blocks
    
    def _parse_term_block(self, lines: List[str]) -> Optional[OBOTerm]:
        """Parse a [Term] block and return an OBOTerm object."""
        term_data = {}
        current_def = ""
        
        for line in lines:
            if not line or line.startswith('!'):
                continue
            
            if ':' not in line:
                # This might be a continuation of the previous line
                if current_def and line.strip():
                    current_def += " " + line.strip()
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Handle different term properties
            if key == 'id':
                term_data['id'] = value
            elif key == 'name':
                term_data['name'] = value
            elif key == 'def':
                # Handle multi-line definitions
                current_def = value
                # Parse definition (remove quotes and extract)
                def_match = re.match(r'"([^"]*)"', current_def)
                if def_match:
                    term_data['definition'] = def_match.group(1)
                else:
                    # Handle cases where definition might span multiple lines or be malformed
                    # Remove leading quote if present
                    if current_def.startswith('"'):
                        current_def = current_def[1:]
                    # Find the end quote and references
                    end_quote_match = re.search(r'"(\s*\[.*\])?$', current_def)
                    if end_quote_match:
                        term_data['definition'] = current_def[:end_quote_match.start()]
                    else:
                        term_data['definition'] = current_def
            elif key == 'synonym':
                if 'synonyms' not in term_data:
                    term_data['synonyms'] = []
                # Extract synonym text from quotes
                syn_match = re.match(r'"([^"]*)"', value)
                if syn_match:
                    synonym_text = syn_match.group(1).strip()
                    if synonym_text:  # Only add non-empty synonyms
                        term_data['synonyms'].append(synonym_text)
            elif key == 'is_obsolete':
                term_data['is_obsolete'] = value.lower() == 'true'
        
        # Create OBOTerm object if we have at least an ID
        if 'id' in term_data:
            return OBOTerm(**term_data)
        
        return None
    
    def save_json(self, output_file: str, indent: int = 2) -> None:
        """
        Save the parsed terms to a JSON file.
        
        Args:
            output_file: Path to the output JSON file
            indent: JSON indentation (None for compact format)
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.terms, f, indent=indent, ensure_ascii=False)
            
            print(f"\n✓ Successfully saved {len(self.terms)} terms to {output_file}")
            self._print_stats()
            
        except Exception as e:
            print(f"✗ Error saving JSON file: {e}")
    
    def save_compact_json(self, output_file: str) -> None:
        """Save the parsed terms to a compact JSON file (no indentation)."""
        self.save_json(output_file, indent=None)
    
    
    def _print_stats(self) -> None:
        """Print processing statistics."""
        print(f"\nProcessing Statistics:")
        print(f"  Files processed: {self.stats['files_processed']}")
        print(f"  Total terms: {self.stats['total_terms']}")
        print(f"  Obsolete terms: {self.stats['obsolete_terms']}")
        print(f"  Active terms: {self.stats['total_terms'] - self.stats['obsolete_terms']}")
        
        # Count terms by prefix
        prefix_counts = {}
        for term_id in self.terms.keys():
            if ':' in term_id:
                prefix = term_id.split(':')[0]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        if prefix_counts:
            print(f"\nTerms by ontology:")
            for prefix, count in sorted(prefix_counts.items()):
                print(f"  {prefix}: {count} terms")
    
    def print_sample_terms(self, n: int = 5) -> None:
        """Print a sample of terms for verification."""
        print(f"\nSample terms:")
        for i, (term_id, term_data) in enumerate(self.terms.items()):
            if i >= n:
                break
            
            synonyms = term_data.get('synonym', '')
            syn_str = f" (synonyms: {synonyms})" if synonyms else ""
            print(f"  {term_id}: {term_data['name']}{syn_str}")
            if term_data['def']:
                def_preview = term_data['def'][:100] + "..." if len(term_data['def']) > 100 else term_data['def']
                print(f"    Definition: {def_preview}")
            print()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Convert OBO ontology files to JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python obo_to_json.py cl.obo -o cl_terms.json
  python obo_to_json.py cl.obo uberon-basic.obo -o combined_ontologies.json
  python obo_to_json.py *.obo -o all_ontologies.json --compact
        """
    )
    
    parser.add_argument('files', nargs='+', help='OBO files to process')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file')
    parser.add_argument('--compact', action='store_true', help='Save JSON in compact format (no indentation)')
    parser.add_argument('--sample', type=int, default=5, help='Number of sample terms to display (default: 5)')
    
    args = parser.parse_args()
    
    try:
        converter = OBOToJSONConverter()
        
        # Process all input files
        converter.parse_files(args.files)
        
        if not converter.terms:
            print("No terms found in the input files.")
            sys.exit(1)
        
        # Show sample terms
        converter.print_sample_terms(args.sample)
        
        # Save to JSON
        if args.compact:
            converter.save_compact_json(args.output)
        else:
            converter.save_json(args.output)
        
        print(f"\n✓ Conversion complete! Output saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n✗ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()