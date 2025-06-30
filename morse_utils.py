"""
Enhanced Morse code utilities for blink-to-text conversion.
Provides advanced pattern recognition and decoding capabilities.
"""

import re
from typing import List, Dict, Tuple, Optional


class MorseCodeProcessor:
    """Advanced Morse code processor with pattern recognition and error correction."""
    
    # Standard International Morse Code
    MORSE_CODE_MAP = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..', 
        '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
        '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----',
        ' ': '/', '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.',
        '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...',
        ':': '---...', ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-',
        '_': '..--.-', '"': '.-..-.', '$': '...-..-', '@': '.--.-.'
    }
    
    # Reverse mapping for decoding
    MORSE_TO_CHAR = {v: k for k, v in MORSE_CODE_MAP.items()}
    
    # Common Morse code patterns for error correction
    COMMON_PATTERNS = {
        'SOS': '...---...',
        'HELP': '......-....-..',
        'OK': '---.-',
        'YES': '-.--.',
        'NO': '-.---',
        'STOP': '...--.----..',
        'END': '.-..-.',
        'START': '...-.-.-'
    }
    
    def __init__(self, 
                 dot_threshold: float = 0.3,
                 dash_threshold: float = 0.6,
                 letter_gap_threshold: float = 1.5,
                 word_gap_threshold: float = 3.0):
        """
        Initialize Morse code processor with timing thresholds.
        
        Args:
            dot_threshold: Maximum duration for a dot (seconds)
            dash_threshold: Minimum duration for a dash (seconds)
            letter_gap_threshold: Minimum gap between letters (seconds)
            word_gap_threshold: Minimum gap between words (seconds)
        """
        self.dot_threshold = dot_threshold
        self.dash_threshold = dash_threshold
        self.letter_gap_threshold = letter_gap_threshold
        self.word_gap_threshold = word_gap_threshold
    
    def analyze_blink_patterns(self, blink_times: List[Dict]) -> Dict:
        """
        Analyze blink patterns and convert to Morse code with advanced timing analysis.
        
        Args:
            blink_times: List of blink dictionaries with 'start', 'end', 'duration'
            
        Returns:
            Dictionary with analysis results including Morse pattern and decoded text
        """
        if not blink_times:
            return {
                'morse_pattern': '',
                'decoded_text': '',
                'confidence': 0.0,
                'analysis': 'No blinks detected'
            }
        
        # Convert blinks to dots and dashes based on duration
        morse_elements = []
        gaps = []
        
        for i, blink in enumerate(blink_times):
            duration = blink['duration']
            
            if duration <= self.dot_threshold:
                morse_elements.append('.')
            elif duration >= self.dash_threshold:
                morse_elements.append('-')
            else:
                # Ambiguous duration - use context or default to dot
                morse_elements.append('.')
            
            # Calculate gap to next blink
            if i < len(blink_times) - 1:
                gap = blink_times[i + 1]['start'] - blink['end']
                gaps.append(gap)
        
        # Group elements into letters and words based on gaps
        morse_letters = self._group_morse_elements(morse_elements, gaps)
        
        # Decode to text
        decoded_text = self._decode_morse_letters(morse_letters)
        
        # Calculate confidence based on valid patterns
        confidence = self._calculate_confidence(morse_letters, decoded_text)
        
        # Generate analysis
        analysis = self._generate_analysis(blink_times, morse_elements, gaps, confidence)
        
        return {
            'morse_pattern': ' '.join(morse_letters),
            'raw_pattern': ''.join(morse_elements),
            'decoded_text': decoded_text,
            'confidence': confidence,
            'analysis': analysis,
            'blink_count': len(blink_times),
            'letter_count': len(morse_letters),
            'timing_stats': self._calculate_timing_stats(blink_times, gaps)
        }
    
    def _group_morse_elements(self, elements: List[str], gaps: List[float]) -> List[str]:
        """Group Morse elements into letters based on gap timing."""
        if not elements:
            return []
        
        letters = []
        current_letter = elements[0]
        
        for i, gap in enumerate(gaps):
            if gap >= self.word_gap_threshold:
                # End of word
                letters.append(current_letter)
                letters.append('/')  # Word separator
                current_letter = elements[i + 1] if i + 1 < len(elements) else ''
            elif gap >= self.letter_gap_threshold:
                # End of letter
                letters.append(current_letter)
                current_letter = elements[i + 1] if i + 1 < len(elements) else ''
            else:
                # Continue current letter
                if i + 1 < len(elements):
                    current_letter += elements[i + 1]
        
        # Add the last letter
        if current_letter:
            letters.append(current_letter)
        
        return letters
    
    def _decode_morse_letters(self, morse_letters: List[str]) -> str:
        """Decode Morse letters to text with error handling."""
        decoded_chars = []
        
        for morse_letter in morse_letters:
            if morse_letter == '/':
                decoded_chars.append(' ')
            elif morse_letter in self.MORSE_TO_CHAR:
                decoded_chars.append(self.MORSE_TO_CHAR[morse_letter])
            elif morse_letter:
                # Try error correction
                corrected = self._attempt_error_correction(morse_letter)
                decoded_chars.append(corrected if corrected else '?')
        
        return ''.join(decoded_chars)
    
    def _attempt_error_correction(self, morse_pattern: str) -> Optional[str]:
        """Attempt to correct common Morse code errors."""
        # Try removing or adding dots/dashes
        variations = [
            morse_pattern[1:],  # Remove first element
            morse_pattern[:-1],  # Remove last element
            '.' + morse_pattern,  # Add dot at start
            morse_pattern + '.',  # Add dot at end
            '-' + morse_pattern,  # Add dash at start
            morse_pattern + '-',  # Add dash at end
        ]
        
        for variation in variations:
            if variation in self.MORSE_TO_CHAR:
                return self.MORSE_TO_CHAR[variation]
        
        return None
    
    def _calculate_confidence(self, morse_letters: List[str], decoded_text: str) -> float:
        """Calculate confidence score based on valid patterns and common words."""
        if not morse_letters:
            return 0.0
        
        valid_letters = sum(1 for letter in morse_letters 
                          if letter in self.MORSE_TO_CHAR or letter == '/')
        total_letters = len([l for l in morse_letters if l != '/'])
        
        if total_letters == 0:
            return 0.0
        
        pattern_confidence = valid_letters / len(morse_letters)
        
        # Bonus for common patterns
        text_upper = decoded_text.upper()
        common_bonus = 0.0
        for pattern, morse in self.COMMON_PATTERNS.items():
            if pattern in text_upper:
                common_bonus += 0.2
        
        return min(1.0, pattern_confidence + common_bonus)
    
    def _calculate_timing_stats(self, blink_times: List[Dict], gaps: List[float]) -> Dict:
        """Calculate timing statistics for analysis."""
        if not blink_times:
            return {}
        
        durations = [blink['duration'] for blink in blink_times]
        
        stats = {
            'avg_blink_duration': sum(durations) / len(durations),
            'min_blink_duration': min(durations),
            'max_blink_duration': max(durations),
            'total_duration': blink_times[-1]['end'] - blink_times[0]['start'],
        }
        
        if gaps:
            stats.update({
                'avg_gap': sum(gaps) / len(gaps),
                'min_gap': min(gaps),
                'max_gap': max(gaps),
            })
        
        return stats
    
    def _generate_analysis(self, blink_times: List[Dict], morse_elements: List[str], 
                          gaps: List[float], confidence: float) -> str:
        """Generate human-readable analysis of the Morse code detection."""
        analysis_parts = []
        
        if not blink_times:
            return "No blinks detected in the video."
        
        analysis_parts.append(f"Detected {len(blink_times)} blinks over {blink_times[-1]['end'] - blink_times[0]['start']:.1f} seconds.")
        
        dots = morse_elements.count('.')
        dashes = morse_elements.count('-')
        analysis_parts.append(f"Pattern: {dots} dots, {dashes} dashes.")
        
        if confidence > 0.8:
            analysis_parts.append("High confidence in Morse code detection.")
        elif confidence > 0.5:
            analysis_parts.append("Moderate confidence - some patterns may be unclear.")
        else:
            analysis_parts.append("Low confidence - timing may not follow standard Morse code.")
        
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            if avg_gap > self.word_gap_threshold:
                analysis_parts.append("Long gaps detected - likely multiple words.")
            elif avg_gap > self.letter_gap_threshold:
                analysis_parts.append("Medium gaps detected - likely multiple letters.")
        
        return " ".join(analysis_parts)
    
    def encode_text_to_morse(self, text: str) -> str:
        """Convert text to Morse code pattern."""
        morse_parts = []
        
        for char in text.upper():
            if char == ' ':
                morse_parts.append('/')
            elif char in self.MORSE_CODE_MAP:
                morse_parts.append(self.MORSE_CODE_MAP[char])
            else:
                morse_parts.append('?')  # Unknown character
        
        return ' '.join(morse_parts)
    
    def get_morse_reference(self) -> Dict[str, str]:
        """Get the complete Morse code reference."""
        return self.MORSE_CODE_MAP.copy()


# Utility functions for easy integration
def analyze_blinks_to_morse(blink_times: List[Dict], 
                           dot_threshold: float = 0.3,
                           dash_threshold: float = 0.6) -> Dict:
    """
    Quick utility function to analyze blinks and convert to Morse code.
    
    Args:
        blink_times: List of blink dictionaries
        dot_threshold: Maximum duration for dots
        dash_threshold: Minimum duration for dashes
        
    Returns:
        Analysis results dictionary
    """
    processor = MorseCodeProcessor(dot_threshold, dash_threshold)
    return processor.analyze_blink_patterns(blink_times)


def text_to_morse(text: str) -> str:
    """Convert text to Morse code pattern."""
    processor = MorseCodeProcessor()
    return processor.encode_text_to_morse(text)


def get_morse_reference() -> Dict[str, str]:
    """Get Morse code reference dictionary."""
    processor = MorseCodeProcessor()
    return processor.get_morse_reference()

