import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Activity,
  TrendingUp,
  Database,
  Brain,
  CheckCircle,
} from 'lucide-react'

export function Dashboard() {
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Trading Dashboard</h1>
        <div className="flex items-center space-x-2">
          <Badge variant="outline">Last Updated: 2 min ago</Badge>
          <Button size="sm">
            <Activity className="h-4 w-4 mr-2" />
            Live View
          </Button>
        </div>
      </div>

      {/* System Status Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">Online</div>
            <p className="text-xs text-muted-foreground">
              All systems operational
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Signals</CardTitle>
            <TrendingUp className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">
              +2 from last hour
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Accuracy</CardTitle>
            <Brain className="h-4 w-4 text-purple-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">87.3%</div>
            <p className="text-xs text-muted-foreground">
              Last 100 predictions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Quality</CardTitle>
            <Database className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">95.2%</div>
            <p className="text-xs text-muted-foreground">
              Real-time data health
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="signals">Recent Signals</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="health">System Health</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Current Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Mode</span>
                  <Badge variant="secondary">HISTORICAL_MODE</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Instruments</span>
                  <span className="text-sm font-medium">1 active</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Timeframes</span>
                  <span className="text-sm font-medium">5m, 15m, 60m</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Indicators</span>
                  <span className="text-sm font-medium">15 enabled</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Start Live Mode
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Brain className="h-4 w-4 mr-2" />
                  Train New Model
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Database className="h-4 w-4 mr-2" />
                  Sync Instruments
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="signals" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Trading Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { time: '14:30', instrument: 'NIFTY 50', signal: 'BUY', confidence: 0.82 },
                  { time: '14:15', instrument: 'NIFTY 50', signal: 'HOLD', confidence: 0.65 },
                  { time: '14:00', instrument: 'NIFTY 50', signal: 'SELL', confidence: 0.78 },
                ].map((signal, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="text-sm font-medium">{signal.time}</div>
                      <div className="text-sm">{signal.instrument}</div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <Badge
                        variant={signal.signal === 'BUY' ? 'default' : signal.signal === 'SELL' ? 'destructive' : 'secondary'}
                      >
                        {signal.signal}
                      </Badge>
                      <div className="text-sm text-muted-foreground">
                        {Math.round(signal.confidence * 100)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Metrics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Accuracy</span>
                    <span className="text-sm font-medium">87.3%</span>
                  </div>
                  <Progress value={87.3} />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Precision</span>
                    <span className="text-sm font-medium">84.1%</span>
                  </div>
                  <Progress value={84.1} />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Recall</span>
                    <span className="text-sm font-medium">89.7%</span>
                  </div>
                  <Progress value={89.7} />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="health" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>System Health Monitoring</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { component: 'Database Connection', status: 'healthy', uptime: '99.9%' },
                  { component: 'Broker API', status: 'healthy', uptime: '98.7%' },
                  { component: 'WebSocket Stream', status: 'healthy', uptime: '99.1%' },
                  { component: 'Model Predictor', status: 'healthy', uptime: '100%' },
                ].map((item, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium">{item.component}</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <Badge variant="outline">{item.uptime}</Badge>
                      <span className="text-sm text-green-600 capitalize">{item.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}