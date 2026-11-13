import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Medical Decision Support — RG24",
  description:
    "Multimodal deep learning that links patient-reported symptoms to chest radiographs for multi-disease prediction and triage. Major Project-I, JUIT.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="site">
          <div className="wrap">
            <h1>AI-Based Medical Decision-Support System</h1>
            <p>
              Multimodal deep learning linking patient-reported symptoms to chest radiographs ·
              Major Project-I (18B19CI791) · Group RG24 · Jaypee University of Information Technology
            </p>
            <div className="badge-row">
              <span className="badge">DDXPlus · 610k patients</span>
              <span className="badge">Open-I · 3,105 radiographs</span>
              <span className="badge">Cross-modal attention</span>
              <span className="badge">Runs in your browser</span>
            </div>
          </div>
        </header>
        <main className="wrap">{children}</main>
      </body>
    </html>
  );
}
