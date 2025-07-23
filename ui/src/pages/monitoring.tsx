import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Activity, ExternalLink, AlertCircle, CheckCircle, Database, Server } from 'lucide-react'

export function Monitoring() {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">System Monitoring</h1>
          <p className="text-muted-foreground mt-1">
            Quick overview and links to detailed monitoring
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" asChild>
            <a href="http://localhost:3000" target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-4 w-4 mr-2" />
              Grafana Dashboard
            </a>
          </Button>
          <Button variant="outline" asChild>
            <a href="http://localhost:9090" target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-4 w-4 mr-2" />
              Prometheus
            </a>
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">System Health</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span className="text-sm text-green-600">Healthy</span>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">0</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Uptime</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">99.9%</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">67%</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Service Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { name: 'Database (TimescaleDB)', status: 'healthy', uptime: '99.9%' },
                { name: 'Broker API', status: 'healthy', uptime: '98.7%' },
                { name: 'WebSocket Stream', status: 'healthy', uptime: '99.1%' },
                { name: 'Model Predictor', status: 'healthy', uptime: '100%' },
                { name: 'Feature Calculator', status: 'healthy', uptime: '99.8%' },
              ].map((service, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="font-medium">{service.name}</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Badge variant="outline">{service.uptime}</Badge>
                    <span className="text-sm text-green-600 capitalize">{service.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">CPU Usage</span>
                  <span className="text-sm font-medium">45%</span>
                </div>
                <Progress value={45} />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Memory Usage</span>
                  <span className="text-sm font-medium">67%</span>
                </div>
                <Progress value={67} />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Disk Usage</span>
                  <span className="text-sm font-medium">23%</span>
                </div>
                <Progress value={23} />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Network I/O</span>
                  <span className="text-sm font-medium">12%</span>
                </div>
                <Progress value={12} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Monitoring Tools</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="p-4 border rounded-lg">
              <div className="flex items-center space-x-3 mb-2">
                <Activity className="h-5 w-5 text-orange-500" />
                <h3 className="font-semibold">Grafana Dashboard</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Comprehensive system metrics, trading performance, and real-time monitoring
              </p>
              <Button variant="outline" asChild>
                <a href="http://localhost:3000" target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Open Dashboard
                </a>
              </Button>
            </div>

            <div className="p-4 border rounded-lg">
              <div className="flex items-center space-x-3 mb-2">
                <Database className="h-5 w-5 text-red-500" />
                <h3 className="font-semibold">Prometheus</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Metrics collection, alerting, and time-series data storage
              </p>
              <Button variant="outline" asChild>
                <a href="http://localhost:9090" target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Open Prometheus
                </a>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-2">
            <Button variant="outline">
              <Activity className="h-4 w-4 mr-2" />
              Refresh Metrics
            </Button>
            <Button variant="outline">
              <AlertCircle className="h-4 w-4 mr-2" />
              View Alerts
            </Button>
            <Button variant="outline">
              <Server className="h-4 w-4 mr-2" />
              System Logs
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}