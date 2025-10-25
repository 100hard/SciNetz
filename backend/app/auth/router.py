"""FastAPI router for authentication endpoints."""
from __future__ import annotations

from typing import AsyncIterator, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.auth.repository import AuthRepository
from backend.app.auth.schemas import (
    AuthUser,
    GoogleConfigResponse,
    GoogleLoginRequest,
    LoginResponse,
    SessionStatusResponse,
    LogoutRequest,
    LogoutResponse,
    RefreshRequest,
    RegistrationResponse,
    RegisterRequest,
    TokenRefreshResponse,
    VerificationResponse,
)
from backend.app.auth.service import AuthService, AuthServiceError
from backend.app.auth.utils import EmailDispatcher, GoogleTokenVerifier, JWTManager
from backend.app.config import AuthConfig

router = APIRouter(prefix="/api/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/google")
oauth2_optional_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/google", auto_error=False
)


def _status_from_reason(reason: str) -> int:
    """Translate service error reasons into HTTP status codes."""

    mapping = {
        "bad_request": status.HTTP_400_BAD_REQUEST,
        "conflict": status.HTTP_409_CONFLICT,
        "unauthorized": status.HTTP_401_UNAUTHORIZED,
        "forbidden": status.HTTP_403_FORBIDDEN,
        "not_found": status.HTTP_404_NOT_FOUND,
        "gone": status.HTTP_410_GONE,
    }
    return mapping.get(reason, status.HTTP_400_BAD_REQUEST)


def get_auth_config(request: Request) -> AuthConfig:
    """Resolve the auth configuration from the application state."""

    return request.app.state.auth_config


async def get_auth_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield an auth database session."""

    session_factory = cast(
        async_sessionmaker[AsyncSession], request.app.state.auth_session_factory
    )
    async with session_factory() as session:
        yield session


def get_jwt_manager(request: Request) -> JWTManager:
    """Return the JWT manager stored on the app state."""

    return request.app.state.jwt_manager


def get_email_dispatcher(request: Request) -> EmailDispatcher:
    """Return the email dispatcher stored on the app state."""

    return request.app.state.email_dispatcher


def get_google_verifier(request: Request) -> GoogleTokenVerifier:
    """Return the Google token verifier stored on the app state."""

    return request.app.state.google_verifier


async def get_auth_service(
    session: AsyncSession = Depends(get_auth_session),
    config: AuthConfig = Depends(get_auth_config),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    dispatcher: EmailDispatcher = Depends(get_email_dispatcher),
    google_verifier: GoogleTokenVerifier = Depends(get_google_verifier),
) -> AuthService:
    """Construct an AuthService for the current request."""

    repository = AuthRepository(session)
    return AuthService(
        config=config,
        repository=repository,
        jwt_manager=jwt_manager,
        email_dispatcher=dispatcher,
        google_verifier=google_verifier,
    )


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    service: AuthService = Depends(get_auth_service),
) -> AuthUser:
    """Validate bearer token and return the authenticated user."""

    try:
        return await service.authenticate(token)
    except AuthServiceError as exc:
        raise HTTPException(status_code=_status_from_reason(exc.reason), detail=str(exc)) from exc


async def get_optional_user(
    token: str | None = Depends(oauth2_optional_scheme),
    service: AuthService = Depends(get_auth_service),
) -> Optional[AuthUser]:
    """Attempt to authenticate bearer token, returning None when absent or invalid."""

    if not token:
        return None
    try:
        return await service.authenticate(token)
    except AuthServiceError as exc:
        if exc.reason in {"unauthorized", "forbidden"}:
            return None
        raise HTTPException(status_code=_status_from_reason(exc.reason), detail=str(exc)) from exc


@router.post("/register", response_model=RegistrationResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    background_tasks: BackgroundTasks,
    service: AuthService = Depends(get_auth_service),
) -> RegistrationResponse:
    """Register a new user account and send verification email."""

    try:
        response, token, expires_at = await service.register_user(payload)
    except AuthServiceError as exc:
        raise HTTPException(status_code=_status_from_reason(exc.reason), detail=str(exc)) from exc
    if token is not None and expires_at is not None:
        background_tasks.add_task(service.send_verification_email, payload.email, token, expires_at)
    return response


@router.post("/google", response_model=LoginResponse)
async def login_with_google(
    payload: GoogleLoginRequest, service: AuthService = Depends(get_auth_service)
) -> LoginResponse:
    """Authenticate a Google credential."""

    try:
        return await service.login_with_google(payload)
    except AuthServiceError as exc:
        raise HTTPException(status_code=_status_from_reason(exc.reason), detail=str(exc)) from exc


@router.get("/google/config", response_model=GoogleConfigResponse)
async def google_config(config: AuthConfig = Depends(get_auth_config)) -> GoogleConfigResponse:
    """Expose Google authentication configuration for the UI."""

    return GoogleConfigResponse(client_ids=list(config.google.client_ids))


@router.post("/token/refresh", response_model=TokenRefreshResponse)
async def refresh_tokens(
    payload: RefreshRequest, service: AuthService = Depends(get_auth_service)
) -> TokenRefreshResponse:
    """Refresh access tokens using a valid refresh token."""

    try:
        return await service.refresh_tokens(payload)
    except AuthServiceError as exc:
        raise HTTPException(status_code=_status_from_reason(exc.reason), detail=str(exc)) from exc


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    payload: LogoutRequest,
    service: AuthService = Depends(get_auth_service),
    current_user: AuthUser = Depends(get_current_user),
) -> LogoutResponse:
    """Revoke a refresh token for the current user."""

    try:
        return await service.logout(payload.refresh_token, user_id=current_user.id)
    except AuthServiceError as exc:
        raise HTTPException(status_code=_status_from_reason(exc.reason), detail=str(exc)) from exc


@router.get("/verify", response_model=VerificationResponse)
async def verify(token: str = Query(..., min_length=1), service: AuthService = Depends(get_auth_service)) -> VerificationResponse:
    """Verify a registration token."""

    try:
        return await service.verify_email(token)
    except AuthServiceError as exc:
        raise HTTPException(status_code=_status_from_reason(exc.reason), detail=str(exc)) from exc


@router.get("/me", response_model=SessionStatusResponse)
async def me(current_user: Optional[AuthUser] = Depends(get_optional_user)) -> SessionStatusResponse:
    """Return authentication status for the current session."""

    if current_user is None:
        return SessionStatusResponse(authenticated=False, user=None)
    return SessionStatusResponse(authenticated=True, user=current_user)


__all__ = [
    "router",
    "get_current_user",
    "get_optional_user",
    "get_auth_service",
    "get_auth_session",
    "get_auth_config",
    "get_google_verifier",
]
