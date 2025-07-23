import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  LayoutDashboard,
  Settings,
  TrendingUp,
  Database,
  Brain,
  Calendar,
  Activity,
  ChevronLeft,
  ChevronRight,
  Trash2,
  BarChart3
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Charts', href: '/charts', icon: BarChart3 },
  { name: 'Configuration', href: '/configuration', icon: Settings },
  { name: 'Indicators', href: '/indicators', icon: TrendingUp },
  { name: 'Instruments', href: '/instruments', icon: Database },
  { name: 'ML Models', href: '/models', icon: Brain },
  { name: 'Calendar', href: '/calendar', icon: Calendar },
  { name: 'Monitoring', href: '/monitoring', icon: Activity },
  { name: 'Data Management', href: '/data-management', icon: Trash2 },
]

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const location = useLocation()

  return (
    <div
      className={cn(
        'flex flex-col bg-card border-r border-border transition-all duration-300',
        isCollapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4">
        {!isCollapsed && (
          <h2 className="text-lg font-semibold text-foreground">
            Trading Control
          </h2>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="h-8 w-8"
        >
          {isCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-2">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                'flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
            >
              <item.icon className="h-4 w-4 flex-shrink-0" />
              {!isCollapsed && <span className="ml-3">{item.name}</span>}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center">
          <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
          {!isCollapsed && (
            <span className="ml-2 text-xs text-muted-foreground">
              System Online
            </span>
          )}
        </div>
      </div>
    </div>
  )
}