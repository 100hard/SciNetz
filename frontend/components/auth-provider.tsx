"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import apiClient from "@/lib/http";
import type {
  AuthUser,
  LoginResponse,
  SessionStatusResponse,
  TokenPair,
  TokenRefreshResponse,
} from "../types/auth";

const AUTH_STORAGE_KEY = "scinets.auth.session";

type AuthStatus = "loading" | "authenticated" | "unauthenticated";

type AuthContextValue = {
  status: AuthStatus;
  user: AuthUser | null;
  accessToken: string | null;
  refreshToken: string | null;
  expiresAt: number | null;
  loginWithGoogle: (credential: string) => Promise<LoginResponse>;
  logout: () => Promise<void>;
  refresh: () => Promise<TokenPair>;
};

interface StoredAuthSession {
  user: AuthUser;
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const readUserAgent = (): string | undefined => {
  if (typeof navigator === "undefined") {
    return undefined;
  }
  return navigator.userAgent;
};

const initialState = {
  status: "loading" as AuthStatus,
  user: null as AuthUser | null,
  tokens: {
    accessToken: null as string | null,
    refreshToken: null as string | null,
    expiresAt: null as number | null,
  },
};

type AuthState = typeof initialState;

const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [state, setState] = useState<AuthState>(initialState);
  const refreshInFlight = useRef<Promise<TokenPair> | null>(null);

  const persistSession = useCallback((user: AuthUser, tokens: TokenPair) => {
    const expiresAt = Date.now() + tokens.expires_in * 1000;
    setState({
      status: "authenticated",
      user,
      tokens: {
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token,
        expiresAt,
      },
    });
    if (typeof window !== "undefined") {
      const stored: StoredAuthSession = {
        user,
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token,
        expiresAt,
      };
      window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(stored));
    }
  }, []);

  const clearSession = useCallback(() => {
    setState({
      status: "unauthenticated",
      user: null,
      tokens: {
        accessToken: null,
        refreshToken: null,
        expiresAt: null,
      },
    });
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
    }
  }, []);

  const refresh = useCallback(async () => {
    const currentUser = state.user;
    const currentRefreshToken = state.tokens.refreshToken;

    if (!currentUser || !currentRefreshToken) {
      throw new Error("No session available to refresh");
    }

    const { data } = await apiClient.post<TokenRefreshResponse>("/api/auth/token/refresh", {
      refresh_token: currentRefreshToken,
      user_agent: readUserAgent(),
    });

    persistSession(currentUser, data.tokens);
    return data.tokens;
  }, [persistSession, state.tokens.refreshToken, state.user]);

  const ensureRefresh = useCallback(() => {
    if (!refreshInFlight.current) {
      refreshInFlight.current = refresh()
        .then((tokens) => {
          return tokens;
        })
        .finally(() => {
          refreshInFlight.current = null;
        });
    }
    return refreshInFlight.current;
  }, [refresh]);

  useEffect(() => {
    const interceptorId = apiClient.interceptors.request.use((config) => {
      if (state.tokens.accessToken) {
        config.headers = config.headers ?? {};
        config.headers.Authorization = `Bearer ${state.tokens.accessToken}`;
      } else if (config.headers?.Authorization) {
        delete config.headers.Authorization;
      }
      return config;
    });

    return () => {
      apiClient.interceptors.request.eject(interceptorId);
    };
  }, [state.tokens.accessToken]);

  useEffect(() => {
    let isMounted = true;

    const responseInterceptorId = apiClient.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (!isMounted) {
          return Promise.reject(error);
        }

        const status = error?.response?.status;
        if (status !== 401) {
          return Promise.reject(error);
        }

        const originalRequest = (error.config ?? {}) as typeof error.config & { _retry?: boolean };
        const requestUrl = originalRequest?.url ?? "";

        if (typeof requestUrl === "string" && requestUrl.includes("/api/auth/token/refresh")) {
          clearSession();
          return Promise.reject(error);
        }

        if (originalRequest._retry) {
          clearSession();
          return Promise.reject(error);
        }
        if (!state.tokens.refreshToken) {
          clearSession();
          return Promise.reject(error);
        }

        originalRequest._retry = true;

        try {
          const tokens = await ensureRefresh();
          originalRequest.headers = originalRequest.headers ?? {};
          originalRequest.headers.Authorization = `Bearer ${tokens.access_token}`;
          return apiClient(originalRequest);
        } catch (refreshError) {
          clearSession();
          return Promise.reject(refreshError);
        }
      },
    );

    return () => {
      isMounted = false;
      apiClient.interceptors.response.eject(responseInterceptorId);
    };
  }, [clearSession, ensureRefresh, state.tokens.refreshToken]);

  useEffect(() => {
    if (
      typeof window === "undefined" ||
      state.status !== "authenticated" ||
      !state.tokens.expiresAt ||
      !state.tokens.refreshToken
    ) {
      return;
    }

    const LEAD_MS = 60_000;
    const delay = Math.max(state.tokens.expiresAt - LEAD_MS - Date.now(), 0);
    const timeoutId = window.setTimeout(() => {
      const pending = ensureRefresh();
      pending?.catch(() => {
        clearSession();
      });
    }, delay);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [clearSession, ensureRefresh, state.status, state.tokens.expiresAt, state.tokens.refreshToken]);

  useEffect(() => {
    let isMounted = true;

    const restoreSession = async () => {
      if (typeof window === "undefined") {
        return;
      }

      const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
      if (!raw) {
        if (isMounted) {
          setState((prev) => ({
            ...prev,
            status: "unauthenticated",
          }));
        }
        return;
      }

      try {
        const stored = JSON.parse(raw) as StoredAuthSession;
        if (!stored.accessToken || !stored.refreshToken || !stored.user) {
          throw new Error("Invalid stored session");
        }

        let accessToken = stored.accessToken;
        let refreshToken = stored.refreshToken;
        let expiresAt = stored.expiresAt ?? 0;

        if (!expiresAt || expiresAt <= Date.now()) {
          const { data } = await apiClient.post<TokenRefreshResponse>(
            "/api/auth/token/refresh",
            {
              refresh_token: refreshToken,
              user_agent: readUserAgent(),
            },
          );
          accessToken = data.tokens.access_token;
          refreshToken = data.tokens.refresh_token;
          expiresAt = Date.now() + data.tokens.expires_in * 1000;
        }

        if (!isMounted) {
          return;
        }

        setState({
          status: "loading",
          user: stored.user,
          tokens: {
            accessToken,
            refreshToken,
            expiresAt,
          },
        });

        if (typeof window !== "undefined") {
          const nextStored: StoredAuthSession = {
            user: stored.user,
            accessToken,
            refreshToken,
            expiresAt,
          };
          window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(nextStored));
        }

        try {
          const { data } = await apiClient.get<SessionStatusResponse>("/api/auth/me");
          if (!isMounted) {
            return;
          }
          if (!data.authenticated || !data.user) {
            clearSession();
            return;
          }
          const remaining = Math.max(Math.floor((expiresAt - Date.now()) / 1000), 1);
          const sessionTokens: TokenPair = {
            access_token: accessToken,
            refresh_token: refreshToken,
            token_type: "bearer",
            expires_in: remaining,
          };
          persistSession(data.user, sessionTokens);
        } catch (error) {
          if (!isMounted) {
            return;
          }
          clearSession();
        }
      } catch (error) {
        if (!isMounted) {
          return;
        }
        clearSession();
      }
    };

    void restoreSession();

    return () => {
      isMounted = false;
    };
  }, [clearSession, persistSession]);

  const loginWithGoogle = useCallback(
    async (credential: string) => {
      const { data } = await apiClient.post<LoginResponse>("/api/auth/google", {
        credential,
        user_agent: readUserAgent(),
      });
      persistSession(data.user, data.tokens);
      return data;
    },
    [persistSession],
  );

  const logout = useCallback(async () => {
    const refreshToken = state.tokens.refreshToken;
    let capturedError: unknown;

    if (refreshToken) {
      try {
        await apiClient.post("/api/auth/logout", { refresh_token: refreshToken });
      } catch (error) {
        capturedError = error;
      }
    }

    clearSession();

    if (capturedError) {
      throw capturedError;
    }
  }, [clearSession, state.tokens.refreshToken]);

  const value = useMemo<AuthContextValue>(
    () => ({
      status: state.status,
      user: state.user,
      accessToken: state.tokens.accessToken,
      refreshToken: state.tokens.refreshToken,
      expiresAt: state.tokens.expiresAt,
      loginWithGoogle,
      logout,
      refresh,
    }),
    [loginWithGoogle, logout, refresh, state.status, state.tokens.accessToken, state.tokens.expiresAt, state.tokens.refreshToken, state.user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextValue => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

export default AuthProvider;
