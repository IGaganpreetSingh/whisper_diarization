from openai import OpenAI
import re
from typing import List, Dict
import time


class TranscriptionProcessor:
    def __init__(self, max_segment_words: int = 1200):
        self.client = OpenAI()
        self.max_segment_words = max_segment_words

    def preprocess_transcription(self, text: str) -> List[Dict]:
        """Split transcription into manageable segments."""
        segments = []
        current_segment = []
        word_count = 0

        speaker_turns = re.split(r"(?=Speaker \d+:)", text)
        speaker_turns = [turn.strip() for turn in speaker_turns if turn.strip()]

        for turn in speaker_turns:
            words = turn.split()
            turn_word_count = len(words)

            if turn_word_count > self.max_segment_words:
                if current_segment:
                    segments.append(
                        {"text": " ".join(current_segment), "word_count": word_count}
                    )
                    current_segment = []
                    word_count = 0

                for i in range(0, turn_word_count, self.max_segment_words):
                    chunk_words = words[i : min(i + self.max_segment_words, len(words))]
                    if i > 0:
                        speaker_match = re.match(r"(Speaker \d+):", turn)
                        if speaker_match:
                            chunk_words.insert(0, f"{speaker_match.group(1)}:")

                    segments.append(
                        {"text": " ".join(chunk_words), "word_count": len(chunk_words)}
                    )

            elif word_count + turn_word_count > self.max_segment_words:
                segments.append(
                    {"text": " ".join(current_segment), "word_count": word_count}
                )
                current_segment = [turn]
                word_count = turn_word_count

            else:
                current_segment.append(turn)
                word_count += turn_word_count

        if current_segment:
            segments.append(
                {"text": " ".join(current_segment), "word_count": word_count}
            )

        return segments

    def process_segment(
        self, segment: Dict, segment_num: int, total_segments: int
    ) -> str:
        """Process a single segment."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Changed from gpt-4o-mini to gpt-4
                messages=[
                    {
                        "role": "system",
                        "content": """You are a highly skilled editor specializing in correcting punctuation and grammar errors in legal transcriptions. Your task is to review and correct the provided transcription from a legal hearing, ensuring accuracy while preserving the original text, meaning, and legal terminology.

                    ### Guidelines:
                    1. **Careful Review**: Read the entire transcription thoroughly. Identify and fix ONLY these specific items:
                        - Missing or incorrect periods, commas, semicolons, and colons
                        - Misuse of quotation marks and possessive apostrophes
                        - Run-on sentences that require proper punctuation for clarity
                
                    2. **Contractions**: Expand all contractions into their full forms (e.g., 'you're' → 'you are', 'I'm' → 'I am', 'it's' → 'it is', etc.).
                
                    3. **Language & Terminology**:  
                        - Preserve legal terminology exactly as provided.  
                        - Convert all American English spellings and terms to British English if found (e.g., 'labor' → 'labour', 'program' → 'programme', 'center' → 'centre').  

                    4. **Speaker Attribution Integrity**:
                        - Each speaker's lines must remain exactly with their original speaker.
                        - Never move text between speakers.
                        - Never combine or split speaker segments.
                        - Each speaker's exact words must be preserved verbatim.
                        - Speaker labels (e.g., "Speaker 1:", "Speaker 2:") must remain unchanged.

                    5. **Ambiguities**:  
                        - Leave ambiguities unchanged if unsure about corrections.

                    6. **Time Formats**: 
                        - Standardize time formats (e.g., '01.06' → '01:06') when explicitly discussing time.
                        
                    7. **Verbatim Requirement**:
                        - The transcription must remain word-for-word identical to the original
                        - Focus solely on punctuation and contraction expansion
                        - Do not add, remove, or modify any words
                        - Do not attempt to improve clarity or formality
                        - Do not correct apparent mistakes in word choice or terminology
                        
                    8. **Prohibited Actions**:
                        - No paraphrasing or rewording
                        - No standardization of spelling variations
                        - No improvement of word choices when its person's name or title
                        - No modification of informal language
                        - No reorganization of speaker statements
                        - No merging or splitting of speaker segments
                        - No deletion or addition of words
                        - No modification of '[indistinct]' markers
                        - No truncation or summarization
                        
                    9. **Formatting Requirements**:
                        - Add one blank line (empty line) after each speaker's complete statement
                        - This blank line must appear before the next speaker's label
                        - Maintain consistent spacing throughout the document

                    Your sole focus is on punctuation correction and contraction expansion while maintaining absolute verbatim preservation of the original spoken words.""",
                    },
                    {
                        "role": "user",
                        "content": f"Correct this transcription segment ({segment_num}/{total_segments}):\n\n{segment['text']}",
                    },
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error processing segment {segment_num}: {str(e)}")
            # Return original text if processing fails
            return segment["text"]

    def process_transcription(self, transcription: str) -> str:
        """Process entire transcription and save as fixed_transcription.txt."""
        print("Preprocessing transcription...")
        segments = self.preprocess_transcription(transcription)
        total_segments = len(segments)

        print(f"\nProcessing {total_segments} segments...")
        corrected_segments = []

        for i, segment in enumerate(segments, 1):
            print(f"Processing segment {i}/{total_segments}")

            # Process with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                corrected = self.process_segment(segment, i, total_segments)
                if corrected:
                    corrected_segments.append(corrected)
                    break
                elif attempt < max_retries - 1:
                    print(
                        f"Retrying segment {i} (attempt {attempt + 2}/{max_retries})..."
                    )
                    time.sleep(2)

            # If all retries failed, use original text
            if len(corrected_segments) < i:
                corrected_segments.append(segment["text"])

        # Combine all segments and save
        final_text = "\n".join(corrected_segments)

        print("\nPunctuation processing complete!")
        return final_text


def main():
    processor = TranscriptionProcessor()

    # Read input transcription
    input_file = "1.txt"  # Changed to match your input file
    print(f"Reading transcription from: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        transcription = f.read()

    # Process and save
    processor.process_transcription(transcription)


if __name__ == "__main__":
    main()
