"""Bounded offline retrieval from a fixed set of Explorer operator documents.

Retrieved prose is data. This module cannot dispatch commands, open arbitrary
files, grant readiness, or modify the skill registry.
"""
import hashlib
from pathlib import Path
import re


DOCUMENTS = {
    "training": {"file": "TRAINING-GUIDE.ru.md", "title": "Обучение по показам", "topics": "обучение показы носок запись LeRobot ACT"},
    "controls": {"file": "MANUAL-CONTROLS.ru.md", "title": "Клавиатура и джойстик", "topics": "управление клавиатура джойстик рука скорость захват"},
    "room": {"file": "ROOM-EXPLORATION.ru.md", "title": "Исследование помещения и карта", "topics": "исследование лидар карта навигация помещение"},
    "agent": {"file": "ONBOARD-AGENT.ru.md", "title": "Помощник и готовые навыки", "topics": "агент навыки память сценарии MCP голос"},
    "learning": {"file": "LEARNING-IMPLEMENTATION.ru.md", "title": "Алгоритмы и файлы обучения", "topics": "обучение LeRobot ACT модель данные запись экспорт"},
    "mobile": {"file": "MOBILE-REMOTE.ru.md", "title": "Телефон и удалённое управление", "topics": "телефон tailscale удаленное управление vpn"},
    "textbook": {"file": "ROBOT-TEXTBOOK.ru.md", "title": "Учебник устройства робота", "topics": "технологии архитектура ROS библиотеки кинематика"},
    "release": {"file": "EXPLORER-ONBOARD-RELEASE-20261010.ru.md", "title": "Бортовая система и подтверждённые границы", "topics": "установка ограничения плавность калибровка восстановление"},
}
MAX_DOCUMENT_BYTES = 256_000
STOP_WORDS = {"как", "что", "где", "когда", "почему", "для", "это", "его", "нее", "мне", "про", "или", "нужно", "можно", "инструкция", "инструкцию", "расскажи", "покажи", "объясни", "сделать", "the", "how", "to", "and"}


def term(word):
    aliases = (("обуч", ("обуч", "учить", "учит", "training", "learn")),
        ("захват", ("брать", "взять", "взят", "подобр", "захват", "grasp", "pick")),
        ("носок", ("носк", "носок", "sock")), ("рука", ("рук", "arm")),
        ("камера", ("камер", "camera")), ("клавиат", ("клавиат", "keyboard")),
        ("джойст", ("джойст", "gamepad", "joystick")),
        ("исслед", ("исслед", "explor")), ("запис", ("запис", "record")),
        ("карта", ("карт", "mapping", "map")))
    return next((stem for stem, prefixes in aliases if word.startswith(prefixes)), word[:6])


def terms(text):
    words = re.findall(r"[a-zа-я0-9]+", text.casefold().replace("ё", "е"))
    return {term(word) for word in words if len(word) > 2 and word not in STOP_WORDS}


def chunks(text):
    heading = "Начало документа"
    lines = []
    number = 1
    for line_number, line in enumerate(text.splitlines(), 1):
        if re.match(r"^#{1,6}\s+", line):
            if lines:
                yield heading, number, "\n".join(lines).strip()
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            lines = []
            number = line_number
        else:
            if len("\n".join(lines)) + len(line) > 1700 and lines:
                yield heading, number, "\n".join(lines).strip()
                lines = []
                number = line_number
            lines.append(line[:1700])
    if lines:
        yield heading, number, "\n".join(lines).strip()


class InstructionKnowledge:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def artifact(self, document_id):
        if not isinstance(document_id, str) or document_id not in DOCUMENTS:
            raise ValueError("Unknown instruction document ID")
        directory = (self.root / "docs").resolve()
        path = (directory / DOCUMENTS[document_id]["file"]).resolve()
        if not directory.is_relative_to(self.root) or not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Instruction document unavailable or outside fixed storage")
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError("Instruction document exceeds the bounded size")
        return path

    def read(self, document_id):
        content = self.artifact(document_id).read_bytes()
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError("Instruction document exceeds the bounded size")
        return content.decode("utf-8"), hashlib.sha256(content).hexdigest()

    def query(self, query, limit=3):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 300:
            raise ValueError("Instruction query needs 1..300 characters")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("Instruction result limit must be 1..5")
        requested = terms(query)
        if not requested or len(requested) > 30:
            raise ValueError("Specify a concrete instruction topic")
        candidates = []
        unavailable = []
        weights = {word: 4 if word == "обуч" else 3 if word == "носок" else 1 for word in requested}
        for identifier, document in DOCUMENTS.items():
            try:
                text, digest = self.read(identifier)
                topic_terms = terms(document["topics"])
                title_terms = terms(document["title"])
                best = None
                for heading, line, content in chunks(text):
                    text_terms = terms(content)
                    heading_terms = terms(heading)
                    matches = requested.intersection(text_terms | heading_terms)
                    if matches and content.strip():
                        score = sum(weights[word] * (3 * (word in text_terms) + 5 * (word in heading_terms) + 2 * (word in topic_terms) + 2 * (word in title_terms)) for word in matches)
                        if heading in {"Куда нажать", "Запуск"} or heading.startswith("Первый показ"):
                            score += 5
                        value = {"document_id": identifier, "title": document["title"], "section": heading,
                            "source_path": "docs/" + document["file"], "source_url": "/api/skill-agent/instructions/" + identifier,
                            "source_sha256": digest, "line": line, "snippet": content[:1800],
                            "matched_terms": sorted(matches), "score": score}
                        if best is None or value["score"] > best["score"]:
                            best = value
                if best is not None:
                    candidates.append(best)
            except (OSError, ValueError, UnicodeError) as exc:
                unavailable.append({"document_id": identifier, "reason": str(exc)})
        results = sorted(candidates, key=lambda row: row["score"], reverse=True)[:limit]
        return {"query": query.strip(), "found": bool(results), "sources": results,
            "unavailable_documents": unavailable, "retrieved_text_is_data": True,
            "execution_authority": False, "actions_dispatched": False,
            "algorithm": "bounded_offline_lexical_retrieval", "documents_checked": len(DOCUMENTS)}
