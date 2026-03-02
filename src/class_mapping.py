"""
Class Mapping Utility
=====================
Translate between NASO7Y model classes and project's existing label format.

This allows using NASO7Y pre-trained models with existing datasets annotated
in the project's original format.

NASO7Y Classes (from field_detection.pt):
    0: address, 2: dob, 3: expiry, 4: firstName, 22: issue, 23: job,
    24: lastName, 25: nid, 26: nid_back, 27: photo, 28: poe, 29: serial

Project Classes (original format):
    0: job_title, 1: photo, 2: expiry_date, 3: birth_date, 4: religion,
    5: name, 6: address, 7: national_id, 8: marital_status, 9: gender,
    10: governorate, 11: husband_name, 12: issue_date, 23: serial_number
"""

from typing import Dict, Optional


# NASO7Y class ID → Project class ID
NASO7Y_TO_PROJECT = {
    # Name fields
    4: 5,       # firstName → name (partial match)
    24: 5,      # lastName → name (partial match)
    
    # ID fields
    25: 7,      # nid → national_id
    29: 23,     # serial → serial_number
    
    # Date fields
    2: 3,       # dob → birth_date
    3: 2,       # expiry → expiry_date
    22: 12,     # issue → issue_date
    
    # Other fields
    0: 6,       # address → address
    23: 0,      # job → job_title
    27: 1,      # photo → photo
    
    # NASO7Y fields without project equivalent (mapped to None)
    26: None,   # nid_back → (no equivalent)
    28: None,   # poe → (no equivalent)
}

# Project class ID → NASO7Y class ID
PROJECT_TO_NASO7Y = {
    0: 23,      # job_title → job
    1: 27,      # photo → photo
    2: 3,       # expiry_date → expiry
    3: 2,       # birth_date → dob
    5: 4,       # name → firstName (approximate)
    6: 0,       # address → address
    7: 25,      # national_id → nid
    12: 22,     # issue_date → issue
    23: 29,     # serial_number → serial
    
    # Project fields without NASO7Y equivalent
    4: None,    # religion → (no equivalent)
    8: None,    # marital_status → (no equivalent)
    9: None,    # gender → (no equivalent)
    10: None,   # governorate → (no equivalent)
    11: None,   # husband_name → (no equivalent)
}

# NASO7Y class name → Project class name
NASO7Y_NAMES_TO_PROJECT = {
    "firstName": "name",
    "lastName": "name",
    "nid": "national_id",
    "serial": "serial_number",
    "dob": "birth_date",
    "expiry": "expiry_date",
    "issue": "issue_date",
    "address": "address",
    "job": "job_title",
    "photo": "photo",
    "nid_back": None,
    "poe": None,
}

# Project class name → NASO7Y class name
PROJECT_NAMES_TO_NASO7Y = {
    "job_title": "job",
    "photo": "photo",
    "expiry_date": "expiry",
    "birth_date": "dob",
    "name": "firstName",  # Approximate
    "address": "address",
    "national_id": "nid",
    "issue_date": "issue",
    "serial_number": "serial",
    "religion": None,
    "marital_status": None,
    "gender": None,
    "governorate": None,
    "husband_name": None,
}


def translate_class_id(class_id: int, from_format: str, to_format: str) -> Optional[int]:
    """
    Translate a class ID between formats.
    
    Args:
        class_id: Original class ID
        from_format: Source format ('naso7y' or 'project')
        to_format: Target format ('naso7y' or 'project')
        
    Returns:
        Translated class ID, or None if no equivalent exists
    """
    if from_format == "naso7y" and to_format == "project":
        return NASO7Y_TO_PROJECT.get(class_id)
    elif from_format == "project" and to_format == "naso7y":
        return PROJECT_TO_NASO7Y.get(class_id)
    else:
        return class_id


def translate_class_name(class_name: str, from_format: str, to_format: str) -> Optional[str]:
    """
    Translate a class name between formats.
    
    Args:
        class_name: Original class name
        from_format: Source format ('naso7y' or 'project')
        to_format: Target format ('naso7y' or 'project')
        
    Returns:
        Translated class name, or None if no equivalent exists
    """
    if from_format == "naso7y" and to_format == "project":
        return NASO7Y_NAMES_TO_PROJECT.get(class_name)
    elif from_format == "project" and to_format == "naso7y":
        return PROJECT_NAMES_TO_NASO7Y.get(class_name)
    else:
        return class_name


def get_naso7y_valid_classes() -> Dict[int, str]:
    """
    Get NASO7Y valid field classes (excluding invalid_* classes).
    
    Returns:
        Dict mapping class_id → class_name for valid fields only
    """
    return {
        0: "address",
        2: "dob",
        3: "expiry",
        4: "firstName",
        22: "issue",
        23: "job",
        24: "lastName",
        25: "nid",
        26: "nid_back",
        27: "photo",
        28: "poe",
        29: "serial",
    }


def get_project_valid_classes() -> Dict[int, str]:
    """
    Get project's original valid field classes.
    
    Returns:
        Dict mapping class_id → class_name for valid text fields
    """
    return {
        0: "job_title",
        2: "expiry_date",
        3: "birth_date",
        5: "name",
        6: "address",
        7: "national_id",
        12: "issue_date",
        23: "serial_number",
    }


def merge_detections(naso7y_detections: list, project_detections: list) -> dict:
    """
    Merge detections from both NASO7Y and project formats.
    Prefers project format when available, falls back to NASO7Y.
    
    Args:
        naso7y_detections: List of detections from NASO7Y model
        project_detections: List of detections from project labels
        
    Returns:
        Dict mapping field_name → detection
    """
    merged = {}
    
    # Add project detections first (preferred)
    for det in project_detections:
        field_name = det.get('class_name')
        if field_name and field_name in get_project_valid_classes().values():
            merged[field_name] = det
    
    # Add NASO7Y detections (translated)
    for det in naso7y_detections:
        naso7y_name = det.get('class_name')
        project_name = translate_class_name(naso7y_name, 'naso7y', 'project')
        
        if project_name and project_name not in merged:
            # Create translated detection
            translated_det = det.copy()
            translated_det['class_name'] = project_name
            merged[project_name] = translated_det
    
    return merged


class ClassMapper:
    """
    Utility class for converting between NASO7Y and project formats.
    
    Usage:
        mapper = ClassMapper()
        
        # Convert NASO7Y detection to project format
        proj_class = mapper.naso7y_to_project(25)  # Returns 7 (national_id)
        
        # Convert project detection to NASO7Y format
        naso7y_class = mapper.project_to_naso7y(7)  # Returns 25 (nid)
        
        # Check if classes are compatible
        if mapper.has_equivalent(25, 'naso7y'):
            print("Has equivalent in project format")
    """
    
    def __init__(self):
        self.naso7y_to_project = NASO7Y_TO_PROJECT
        self.project_to_naso7y = PROJECT_TO_NASO7Y
    
    def naso7y_to_project_id(self, naso7y_id: int) -> Optional[int]:
        """Convert NASO7Y class ID to project class ID."""
        return self.naso7y_to_project.get(naso7y_id)
    
    def project_to_naso7y_id(self, project_id: int) -> Optional[int]:
        """Convert project class ID to NASO7Y class ID."""
        return self.project_to_naso7y.get(project_id)
    
    def naso7y_to_project_name(self, naso7y_name: str) -> Optional[str]:
        """Convert NASO7Y class name to project class name."""
        return NASO7Y_NAMES_TO_PROJECT.get(naso7y_name)
    
    def project_to_naso7y_name(self, project_name: str) -> Optional[str]:
        """Convert project class name to NASO7Y class name."""
        return PROJECT_NAMES_TO_NASO7Y.get(project_name)
    
    def has_equivalent(self, class_id: int, format: str) -> bool:
        """Check if a class has an equivalent in the other format."""
        if format == 'naso7y':
            return self.naso7y_to_project.get(class_id) is not None
        elif format == 'project':
            return self.project_to_naso7y.get(class_id) is not None
        return False
    
    def get_unmapped_classes(self, format: str) -> list:
        """Get list of classes without equivalents in the other format."""
        if format == 'naso7y':
            return [k for k, v in self.naso7y_to_project.items() if v is None]
        elif format == 'project':
            return [k for k, v in self.project_to_naso7y.items() if v is None]
        return []


# Singleton instance
_mapper = ClassMapper()


def get_mapper() -> ClassMapper:
    """Get the singleton ClassMapper instance."""
    return _mapper
