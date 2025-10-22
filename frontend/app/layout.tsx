import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Inter } from "next/font/google";
import { Toaster } from "sonner";

import AuthProvider from "../components/auth-provider";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "SciNets Dashboard",
  description: "Operational console for the SciNets research knowledge graph.",
};

const RootLayout = ({ children }: { children: ReactNode }) => {
  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen bg-background text-foreground antialiased`}>
        <AuthProvider>
          <>
            {children}
            <Toaster richColors position="top-right" closeButton />
          </>
        </AuthProvider>
      </body>
    </html>
  );
};

export default RootLayout;
