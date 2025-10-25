export interface AuthUser {
  id: string;
  email: string;
  is_verified: boolean;
  role: "user";
  created_at: string;
  updated_at: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface RegistrationResponse {
  message: string;
  user: AuthUser;
  requires_verification: boolean;
}

export interface LoginResponse {
  message: string;
  user: AuthUser;
  tokens: TokenPair;
}

export interface SessionStatusResponse {
  authenticated: boolean;
  user: AuthUser | null;
}

export interface TokenRefreshResponse {
  message: string;
  tokens: TokenPair;
}

export interface VerificationResponse {
  message: string;
}

export interface LogoutResponse {
  message: string;
}
