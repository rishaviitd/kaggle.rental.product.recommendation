from datetime import datetime

from sqlalchemy import DateTime, Index, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    client_id: Mapped[str] = mapped_column(Text, primary_key=True)
    visit_id: Mapped[str] = mapped_column(Text, nullable=False)
    date_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class UserBrowsingEvent(Base):
    __tablename__ = "user_browsing_events"
    __table_args__ = (
        Index(
            "ix_user_browsing_events_client_time",
            "client_id",
            "date_time",
        ),
    )

    event_id: Mapped[str] = mapped_column(Text, primary_key=True)
    client_id: Mapped[str] = mapped_column(Text, nullable=False)
    date_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    page_type: Mapped[str | None] = mapped_column(Text)
    slug: Mapped[str | None] = mapped_column(Text)
