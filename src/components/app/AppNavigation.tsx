import { Link, useLocation } from "react-router-dom";
import { Code2, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { mockUser } from "@/data/mockData";

const navLinks = [
  { name: "Dashboard", path: "/dashboard" },
  { name: "Repositories", path: "/repositories" },
  { name: "Reviews", path: "/reviews" },
  { name: "Analytics", path: "/analytics" },
  { name: "Settings", path: "/settings" },
];

export const AppNavigation = () => {
  const location = useLocation();

  return (
    <nav className="sticky top-0 z-50 bg-card/95 backdrop-blur-lg border-b border-border">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link to="/dashboard" className="flex items-center gap-2">
            <div className="p-1.5 bg-gradient-primary rounded-lg">
              <Code2 className="w-5 h-5 text-background" />
            </div>
            <span className="text-lg font-bold text-foreground">IntelliCode Review</span>
          </Link>

          <div className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  location.pathname === link.path
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                }`}
              >
                {link.name}
              </Link>
            ))}
          </div>
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-gradient-primary flex items-center justify-center text-sm font-semibold text-background">
                {mockUser.avatar}
              </div>
              <span className="hidden sm:inline text-sm">{mockUser.name}</span>
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem>Profile</DropdownMenuItem>
            <DropdownMenuItem>Settings</DropdownMenuItem>
            <DropdownMenuItem>
              <Link to="/">Sign Out</Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </nav>
  );
};
