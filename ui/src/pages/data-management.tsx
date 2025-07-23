import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { toast } from '@/hooks/use-toast'
import { useMutation } from '@tanstack/react-query'
import { Database, Trash2, AlertTriangle } from 'lucide-react'

const truncateTables = async (tables: string[]) => {
  const response = await fetch('/api/data/truncate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(tables),
  })
  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || 'Failed to truncate tables')
  }
  return response.json()
}

const TRUNCATABLE_TABLES = [
    "ohlcv_1min",
    "ohlcv_5min",
    "ohlcv_15min",
    "ohlcv_60min",
    "features",
    "engineered_features",
    "cross_asset_features",
    "labels",
    "signals",
    "feature_scores",
    "feature_selection_history",
]

export function DataManagement() {
  const [confirmation, setConfirmation] = useState('')
  const [tableToTruncate, setTableToTruncate] = useState<string | null>(null)

  const truncateMutation = useMutation({ 
    mutationFn: truncateTables, 
    onSuccess: (data) => {
      toast({ title: 'Tables Truncated', description: data.message })
      setTableToTruncate(null)
      setConfirmation('')
    },
    onError: (error: Error) => {
      toast({ title: 'Truncate Failed', description: error.message, variant: 'destructive' })
    }
  })

  const handleTruncate = () => {
    if (tableToTruncate && confirmation === tableToTruncate) {
      truncateMutation.mutate([tableToTruncate])
    } else {
      toast({ title: 'Confirmation Mismatch', description: 'Please type the table name correctly to confirm.', variant: 'destructive' })
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Data Management</h1>
          <p className="text-muted-foreground mt-1">
            Manage your application's data. These are destructive operations.
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Database className="h-5 w-5" />
            <span>Truncate Tables</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {TRUNCATABLE_TABLES.map(table => (
              <Card key={table} className="p-4 flex items-center justify-between">
                <span className="font-mono text-sm">{table}</span>
                <Button variant="destructive" onClick={() => setTableToTruncate(table)}>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Truncate
                </Button>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {tableToTruncate && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5 text-destructive" />
                <span>Confirm Truncate</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                This is a destructive action and cannot be undone. 
                You are about to delete all data from the <span className="font-bold font-mono">{tableToTruncate}</span> table.
              </p>
              <p>
                Please type <span className="font-bold font-mono">{tableToTruncate}</span> to confirm.
              </p>
              <div className="space-y-2">
                <Label htmlFor="confirm-truncate">Confirm Table Name</Label>
                <Input
                  id="confirm-truncate"
                  value={confirmation}
                  onChange={(e) => setConfirmation(e.target.value)}
                  autoFocus
                />
              </div>
              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setTableToTruncate(null)}>
                  Cancel
                </Button>
                <Button 
                  variant="destructive" 
                  onClick={handleTruncate} 
                  disabled={truncateMutation.isPending || confirmation !== tableToTruncate}
                >
                  {truncateMutation.isPending ? 'Truncating...' : 'Truncate Table'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
