"""
Data loading system for all CSV files.
Ensures zero hallucination by only using provided data.
"""

import pandas as pd
import math
from pathlib import Path
from typing import Dict, List, Optional
from models import (
    Message, User, Group, BusinessAccount,
    ConversationType, MediaType
)


def safe_text(x):
    """Safe text conversion that handles NaN and None values."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    if pd.isna(x):
        return ""
    return str(x).lower()


class DataLoader:
    """Loads and manages all dataset CSV files."""

    def __init__(self, dataset_path: str = "../dataset"):
        self.dataset_path = Path(dataset_path)
        self._validate_dataset()
        
        # Data storage
        self.messages: Dict[str, Message] = {}
        self.users: Dict[str, User] = {}
        self.groups: Dict[str, Group] = {}
        self.business_accounts: Dict[str, BusinessAccount] = {}
        
        # Additional dataframes for historical analysis
        self.message_history_df: Optional[pd.DataFrame] = None
        self.message_events_df: Optional[pd.DataFrame] = None
        self.group_members_df: Optional[pd.DataFrame] = None
        self.user_business_history_df: Optional[pd.DataFrame] = None
        self.images_df: Optional[pd.DataFrame] = None
        self.voice_notes_df: Optional[pd.DataFrame] = None
        self.daily_notification_summary_df: Optional[pd.DataFrame] = None
        
        self._load_all_data()

    def _validate_dataset(self):
        """Validate that required files exist."""
        required_files = [
            "messages.csv",
            "users.csv", 
            "groups.csv",
            "business_accounts.csv",
            "output.csv"
        ]
        
        for file in required_files:
            file_path = self.dataset_path / file
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

    def _load_all_data(self):
        """Load all CSV files into memory."""
        print("Loading dataset...")
        
        # Load core structured data
        self._load_messages()
        self._load_users()
        self._load_groups()
        self._load_business_accounts()
        
        # Load additional historical data
        self._load_message_history()
        self._load_message_events()
        self._load_group_members()
        self._load_user_business_history()
        self._load_images()
        self._load_voice_notes()
        self._load_daily_notification_summary()
        
        print(f"Loaded {len(self.messages)} messages")
        print(f"Loaded {len(self.users)} users")
        print(f"Loaded {len(self.groups)} groups")
        print(f"Loaded {len(self.business_accounts)} business accounts")

    def _load_messages(self):
        """Load messages.csv."""
        df = pd.read_csv(self.dataset_path / "messages.csv")
        for _, row in df.iterrows():
            # Handle NaN values for media_type
            media_type_value = row.get('media_type', '')
            if pd.isna(media_type_value):
                media_type_value = ''
            
            message = Message(
                message_id=row['message_id'],
                user_id=row['user_id'],
                conversation_type=ConversationType(row['conversation_type']),
                group_id=row.get('group_id', None) if pd.notna(row.get('group_id', None)) else None,
                business_id=row.get('business_id', None) if pd.notna(row.get('business_id', None)) else None,
                sender_user_id=row.get('sender_user_id', None) if pd.notna(row.get('sender_user_id', None)) else None,
                created_at=row['created_at'],
                message_text=row.get('message_text', '') if pd.notna(row.get('message_text', '')) else '',
                media_type=MediaType(str(media_type_value)),
                media_id=row.get('media_id', None) if pd.notna(row.get('media_id', None)) else None,
                forwarded_count=int(row.get('forwarded_count', 0)) if pd.notna(row.get('forwarded_count', 0)) else 0
            )
            self.messages[message.message_id] = message

    def _load_users(self):
        """Load users.csv."""
        df = pd.read_csv(self.dataset_path / "users.csv")
        for _, row in df.iterrows():
            user = User(
                user_id=row['user_id'],
                do_not_disturb_window=row['do_not_disturb_window'],
                messages_opened_30d=int(row['messages_opened_30d']) if pd.notna(row['messages_opened_30d']) else 0,
                messages_replied_30d=int(row['messages_replied_30d']) if pd.notna(row['messages_replied_30d']) else 0,
                notifications_dismissed_30d=int(row['notifications_dismissed_30d']) if pd.notna(row['notifications_dismissed_30d']) else 0,
                messages_reported_30d=int(row['messages_reported_30d']) if pd.notna(row['messages_reported_30d']) else 0
            )
            self.users[user.user_id] = user

    def _load_groups(self):
        """Load groups.csv."""
        df = pd.read_csv(self.dataset_path / "groups.csv")
        for _, row in df.iterrows():
            group = Group(
                group_id=row['group_id'],
                group_name=row['group_name'],
                group_type=row['group_type'],
                member_count=int(row['member_count']) if pd.notna(row['member_count']) else 0,
                admin_count=int(row['admin_count']) if pd.notna(row['admin_count']) else 0,
                created_at=row['created_at'],
                messages_30d=int(row['messages_30d']) if pd.notna(row['messages_30d']) else 0
            )
            self.groups[group.group_id] = group

    def _load_business_accounts(self):
        """Load business_accounts.csv."""
        df = pd.read_csv(self.dataset_path / "business_accounts.csv")
        for _, row in df.iterrows():
            business = BusinessAccount(
                business_id=row['business_id'],
                display_name=row['display_name'],
                brand_name=row['brand_name'],
                category=row['category'],
                verified=bool(row['verified']),
                official_domain=str(row['official_domain']) if pd.notna(row['official_domain']) else '',
                domain_used_by_sender=str(row['domain_used_by_sender']) if pd.notna(row['domain_used_by_sender']) else '',
                account_age_days=int(row['account_age_days']) if pd.notna(row['account_age_days']) else 0,
                messages_sent_30d=int(row['messages_sent_30d']) if pd.notna(row['messages_sent_30d']) else 0,
                user_reports_30d=int(row['user_reports_30d']) if pd.notna(row['user_reports_30d']) else 0,
                domain_used_by_sender_age_days=int(row['domain_used_by_sender_age_days']) if pd.notna(row['domain_used_by_sender_age_days']) else 0
            )
            self.business_accounts[business.business_id] = business

    def _load_message_history(self):
        """Load message_history.csv for historical context."""
        self.message_history_df = pd.read_csv(self.dataset_path / "message_history.csv")

    def _load_message_events(self):
        """Load message_events.csv for user interaction patterns."""
        self.message_events_df = pd.read_csv(self.dataset_path / "message_events.csv")

    def _load_group_members(self):
        """Load group_members.csv for user-group relationships."""
        self.group_members_df = pd.read_csv(self.dataset_path / "group_members.csv")

    def _load_user_business_history(self):
        """Load user_business_history.csv for business relationships."""
        self.user_business_history_df = pd.read_csv(self.dataset_path / "user_business_history.csv")

    def _load_images(self):
        """Load images.csv for media file paths."""
        self.images_df = pd.read_csv(self.dataset_path / "images.csv")

    def _load_voice_notes(self):
        """Load voice_notes.csv for media file paths."""
        self.voice_notes_df = pd.read_csv(self.dataset_path / "voice_notes.csv")

    def _load_daily_notification_summary(self):
        """Load daily_notification_summary.csv for notification load."""
        self.daily_notification_summary_df = pd.read_csv(
            self.dataset_path / "daily_notification_summary.csv"
        )

    # Getter methods for agent access
    def get_message(self, message_id: str) -> Optional[Message]:
        """Get a message by ID."""
        return self.messages.get(message_id)

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self.users.get(user_id)

    def get_group(self, group_id: str) -> Optional[Group]:
        """Get a group by ID."""
        return self.groups.get(group_id)

    def get_business(self, business_id: str) -> Optional[BusinessAccount]:
        """Get a business account by ID."""
        return self.business_accounts.get(business_id)

    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Get historical messages for a user."""
        if self.message_history_df is None:
            return pd.DataFrame()
        return self.message_history_df[self.message_history_df['user_id'] == user_id]

    def get_user_events(self, user_id: str) -> pd.DataFrame:
        """Get message events for a user."""
        if self.message_events_df is None:
            return pd.DataFrame()
        return self.message_events_df[self.message_events_df['user_id'] == user_id]

    def get_business_history(self, user_id: str, business_id: str) -> pd.DataFrame:
        """Get user-business relationship history."""
        if self.user_business_history_df is None:
            return pd.DataFrame()
        return self.user_business_history_df[
            (self.user_business_history_df['user_id'] == user_id) &
            (self.user_business_history_df['business_id'] == business_id)
        ]

    def get_group_membership(self, user_id: str, group_id: str) -> Optional[dict]:
        """Get a user's membership row for a specific group (or None)."""
        if self.group_members_df is None or not group_id:
            return None
        rows = self.group_members_df[
            (self.group_members_df['user_id'] == user_id) &
            (self.group_members_df['group_id'] == group_id)
        ]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def get_media_path(self, media_id: str, media_type: str) -> Optional[str]:
        """Get file path for media content."""
        if media_type == "image" and self.images_df is not None:
            row = self.images_df[self.images_df['image_id'] == media_id]
            if not row.empty:
                return str(self.dataset_path / "media" / "images" / row.iloc[0]['file_path'])
        elif media_type == "voice" and self.voice_notes_df is not None:
            row = self.voice_notes_df[self.voice_notes_df['voice_note_id'] == media_id]
            if not row.empty:
                return str(self.dataset_path / "media" / "audio" / row.iloc[0]['file_path'])
        return None

    def get_all_messages(self) -> List[Message]:
        """Get all messages for processing."""
        return list(self.messages.values())