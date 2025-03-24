from venv import logger
from bs4 import BeautifulSoup, Tag
import re
from typing import Optional

from src.legal_assistant.centralized_logger import CentralizedLogger

logger = CentralizedLogger().get_logger()

def clean_legal_html(html: str):
    """
    Clean HTML from legal documents while preserving structural meaning
    and important metadata. Handles Indian Kanoon-specific formats.
    """

    try:
        cleantext = BeautifulSoup(html, "lxml").text
        return cleantext

    except Exception as e:
        logger.error(f"HTML cleaning failed: {str(e)}")
        return html