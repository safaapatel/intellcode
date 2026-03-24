import { Navigate, useLocation } from "react-router-dom";
import { getSession } from "@/services/auth";

interface Props {
  children: React.ReactNode;
  requireRole?: "admin" | "reviewer" | "developer";
}

export const ProtectedRoute = ({ children, requireRole }: Props) => {
  const session = getSession();
  const location = useLocation();

  if (!session) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (requireRole && session.role !== requireRole && session.role !== "admin") {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};
