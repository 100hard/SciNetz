import type { ReactNode } from "react";

const AuthLayout = ({ children }: { children: ReactNode }) => {
  return (
    <div className="flex min-h-screen items-center justify-center bg-muted/10 px-4 py-12">
      {children}
    </div>
  );
};

export default AuthLayout;
