# Adapted from ai-Apeiria/assistant/text.py with its tested splitting and speech cleanup preserved.
import re


class SentenceBuffer:
    """Incremental splitter; Markdown/code filtering retains state across deltas."""

    def __init__(self, max_chars=100, first_segment_min_chars=0):
        self.pending = ""
        self.max_chars = max_chars
        self.first_segment_min_chars = first_segment_min_chars
        self.first_segment = True

    def feed(self, text, final=False):
        self.pending += text
        result = []
        while self.pending:
            match = re.search(r"[。！？!?；;\n]+[”’\"』」]?|[.](?=\s)", self.pending)
            if self.first_segment and self.first_segment_min_chars:
                clause = re.search(r"[，,：:]", self.pending[self.first_segment_min_chars - 1:self.max_chars])
                if clause:
                    cut = self.first_segment_min_chars - 1 + clause.end()
                    if match is None or cut < match.end():
                        result.append(self.pending[:cut])
                        self.pending = self.pending[cut:]
                        self.first_segment = False
                        continue
            if match and match.end() <= self.max_chars:
                cut = match.end()
            elif len(self.pending) >= self.max_chars:
                cut = self.pending.rfind("，", 0, self.max_chars) + 1
                if not cut:
                    cut = self.max_chars
            elif final:
                cut = len(self.pending)
            else:
                break
            result.append(self.pending[:cut])
            self.pending = self.pending[cut:]
            self.first_segment = False
        return result


class SpeechText:
    """Speech-only transformation; displayed AI text remains exact."""

    def __init__(self, replacements=None):
        self.replacements = replacements or {}
        self.in_code = False
        self.in_think = False

    def convert(self, text):
        output = []
        for part in re.split(r"(```|<think>|</think>)", text):
            if part == "```":
                self.in_code = not self.in_code
            elif part == "<think>":
                self.in_think = True
            elif part == "</think>":
                self.in_think = False
            elif not self.in_code and not self.in_think:
                output.append(part)
        value = "".join(output)
        value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
        value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
        value = re.sub(r"https?://\S+", "", value)
        value = re.sub(r"[*_`#>|~]", "", value)
        for source, target in self.replacements.items():
            value = value.replace(source, target)
        return value.strip() if re.search(r"[\w\u3040-\u30ff\u4e00-\u9fff]", value) else ""
