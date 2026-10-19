from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # MSSQL Settings
    MSSQL_SERVER: str = Field(default="localhost")
    MSSQL_INSTANCE: Optional[str] = Field(default=None)
    MSSQL_PORT: Optional[int] = Field(default=1433)
    MSSQL_DATABASE: str = Field(default="BachirHome_2026")
    MSSQL_DATABASES: Optional[str] = Field(default="BachirHome_2024,BachirHome_2025,BachirHome_2026")
    MSSQL_USER: str = Field(default="sa")
    MSSQL_PASSWORD: str = Field(default="")
    MSSQL_DRIVER: str = Field(default="ODBC Driver 17 for SQL Server")
    MSSQL_TRUST_SERVER_CERTIFICATE: str = Field(default="yes")
    MSSQL_ENCRYPT: str = Field(default="no")

    # Supabase Settings
    SUPABASE_URL: str = Field(default="")
    SUPABASE_KEY: str = Field(default="")
    SUPABASE_SCHEMA: str = Field(default="recosys")
    SUPABASE_DB_URL: Optional[str] = Field(default=None)

    # Gemini AI & Vertex AI Settings
    GCP_PROJECT: str = Field(default="gen-lang-client-0505999827")
    GCP_LOCATION: str = Field(default="us-central1")
    GEMINI_USE_VERTEX: bool = Field(default=True)
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_MODEL_REASONING: str = Field(default="gemini-2.5-flash")
    GEMINI_MODEL_EMBEDDING: str = Field(default="text-embedding-004")
    GEMINI_MODEL_VISUAL_EMBEDDING: str = Field(default="gemini-embedding-2-preview")
    VISUAL_EMBEDDING_DIM: int = Field(default=768)

    # ML & Latent Space Parameters
    EMBEDDING_DIM: int = Field(default=64)
    ITEM2VEC_WINDOW: int = Field(default=5)
    ITEM2VEC_MIN_COUNT: int = Field(default=1)
    ITEM2VEC_EPOCHS: int = Field(default=30)
    
    # Recommendation Parameters
    ANN_TOP_K: int = Field(default=20)
    SEASONAL_BOOST_ALPHA: float = Field(default=0.3)
    TIME_DECAY_HALF_LIFE_DAYS: float = Field(default=30.0)

    # Inventory & Pricing Parameters
    DEFAULT_SERVICE_LEVEL_Z: float = Field(default=1.65)  # 95% service level
    DEFAULT_LEAD_TIME_DAYS: float = Field(default=7.0)

    @property
    def database_list(self) -> list:
        if self.MSSQL_DATABASES:
            return [d.strip() for d in self.MSSQL_DATABASES.split(",") if d.strip()]
        return [self.MSSQL_DATABASE]

    def get_mssql_connection_string(self, database_name: Optional[str] = None) -> str:
        db = database_name or self.MSSQL_DATABASE
        server = f"{self.MSSQL_SERVER}\\{self.MSSQL_INSTANCE}" if self.MSSQL_INSTANCE else self.MSSQL_SERVER
        if self.MSSQL_PORT and not self.MSSQL_INSTANCE and "\\" not in server:
            server = f"{server},{self.MSSQL_PORT}"
        return (
            f"DRIVER={{{self.MSSQL_DRIVER}}};"
            f"SERVER={server};"
            f"DATABASE={db};"
            f"UID={self.MSSQL_USER};"
            f"PWD={self.MSSQL_PASSWORD};"
            f"Encrypt={self.MSSQL_ENCRYPT};"
            f"TrustServerCertificate={self.MSSQL_TRUST_SERVER_CERTIFICATE};"
        )

    @property
    def mssql_connection_string(self) -> str:
        return self.get_mssql_connection_string(self.MSSQL_DATABASE)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
