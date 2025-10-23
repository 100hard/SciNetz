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
      <head>
        {/* ✅ Expose Google Client ID at runtime so the login page can access it */}
        {process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID && (
          <script
            dangerouslySetInnerHTML={{
              __html: `
                window.NEXT_PUBLIC_GOOGLE_CLIENT_ID = "${process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID}";
              `,
            }}
          />
        )}
        <meta name="google-client-id" content={process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || ""} />
      </head>
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
