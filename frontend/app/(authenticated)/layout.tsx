import type { ReactNode } from "react";

import AuthGuard from "../../components/auth-guard";
import Header from "../../components/header";
import Sidebar from "../../components/sidebar";

const AuthenticatedLayout = ({ children }: { children: ReactNode }) => {
  return (
    <AuthGuard>
      <div className="flex min-h-screen w-full bg-muted/10">
        <Sidebar />
        <div className="flex flex-1 flex-col">
          <Header />
          <main className="flex-1 overflow-y-auto bg-background p-6">{children}</main>
        </div>
      </div>
    </AuthGuard>
  );
};

export default AuthenticatedLayout;
