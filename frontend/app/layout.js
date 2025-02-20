import "./globals.css";

export const metadata = {
  title: "Image Caption Generator",
  description: "Generate captions for images using AI",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
