import React, { useState, useEffect } from 'react';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  TrendingUp, 
  Users, 
  DollarSign,
  Activity,
  Eye,
  Download,
  Search,
  Filter,
  RefreshCw,
  Settings,
  Bell,
  LogOut,
  BarChart3,
  PieChart,
  Calendar,
  Clock
} from 'lucide-react';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterRisk, setFilterRisk] = useState('all');
  const [selectedTransaction, setSelectedTransaction] = useState(null);

  // Mock data for demonstration
  const mockTransactions = [
    {
      transaction_id: 'TX001',
      amount: 5000,
      sender: 'John Doe',
      receiver: 'Jane Smith',
      fraud_score: 0.85,
      risk_level: 'High',
      final_flag: true,
      timestamp: '2024-01-15 14:30:00',
      reasons: ['High amount', 'Unusual time']
    },
    {
      transaction_id: 'TX002',
      amount: 250,
      sender: 'Alice Johnson',
      receiver: 'Bob Wilson',
      fraud_score: 0.35,
      risk_level: 'Low',
      final_flag: false,
      timestamp: '2024-01-15 14:28:00',
      reasons: []
    },
    {
      transaction_id: 'TX003',
      amount: 1500,
      sender: 'Charlie Brown',
      receiver: 'Diana Ross',
      fraud_score: 0.65,
      risk_level: 'Medium',
      final_flag: false,
      timestamp: '2024-01-15 14:25:00',
      reasons: ['New receiver']
    }
  ];

  const stats = {
    totalTransactions: 1247,
    fraudDetected: 89,
    preventedLoss: 234500,
    accuracy: 94.2
  };

  useEffect(() => {
    setTransactions(mockTransactions);
  }, []);

  const getRiskColor = (level) => {
    switch (level) {
      case 'High': return 'text-red-600 bg-red-50';
      case 'Medium': return 'text-yellow-600 bg-yellow-50';
      case 'Low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusIcon = (flagged) => {
    return flagged ? (
      <XCircle className="w-5 h-5 text-red-500" />
    ) : (
      <CheckCircle className="w-5 h-5 text-green-500" />
    );
  };

  const filteredTransactions = transactions.filter(tx => {
    const matchesSearch = tx.transaction_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         tx.sender.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         tx.receiver.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterRisk === 'all' || tx.risk_level.toLowerCase() === filterRisk.toLowerCase();
    return matchesSearch && matchesFilter;
  });

  const handleTransactionClick = async (transaction) => {
    setSelectedTransaction(transaction);
  };

  const StatCard = ({ title, value, icon: Icon, color = 'blue' }) => (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
        </div>
        <div className={`p-3 rounded-lg bg-${color}-100`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
    </div>
  );

  const TransactionModal = ({ transaction, onClose }) => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Transaction Details</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
            <XCircle className="w-6 h-6" />
          </button>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-gray-500">Transaction ID</label>
              <p className="font-medium">{transaction.transaction_id}</p>
            </div>
            <div>
              <label className="text-sm text-gray-500">Amount</label>
              <p className="font-medium">${transaction.amount.toLocaleString()}</p>
            </div>
            <div>
              <label className="text-sm text-gray-500">Sender</label>
              <p className="font-medium">{transaction.sender}</p>
            </div>
            <div>
              <label className="text-sm text-gray-500">Receiver</label>
              <p className="font-medium">{transaction.receiver}</p>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-gray-500">Fraud Score</label>
              <p className="font-medium">{(transaction.fraud_score * 100).toFixed(1)}%</p>
            </div>
            <div>
              <label className="text-sm text-gray-500">Risk Level</label>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(transaction.risk_level)}`}>
                {transaction.risk_level}
              </span>
            </div>
          </div>
          
          <div>
            <label className="text-sm text-gray-500">Timestamp</label>
            <p className="font-medium">{transaction.timestamp}</p>
          </div>
          
          {transaction.reasons.length > 0 && (
            <div>
              <label className="text-sm text-gray-500">Risk Indicators</label>
              <div className="flex flex-wrap gap-2 mt-1">
                {transaction.reasons.map((reason, idx) => (
                  <span key={idx} className="px-2 py-1 bg-red-100 text-red-800 rounded text-xs">
                    {reason}
                  </span>
                ))}
              </div>
            </div>
          )}
          
          <div className="flex gap-2 mt-6">
            <button className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors">
              <Download className="w-4 h-4 inline mr-2" />
              Download Report
            </button>
            <button className="flex-1 bg-gray-100 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-200 transition-colors">
              <Eye className="w-4 h-4 inline mr-2" />
              View SHAP Analysis
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-xl font-bold text-gray-900">Fraud Detection Dashboard</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg">
                <Bell className="w-5 h-5" />
              </button>
              <button className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg">
                <Settings className="w-5 h-5" />
              </button>
              <button className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg">
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b">
        <div className="px-6">
          <div className="flex space-x-8">
            {[
              { id: 'overview', label: 'Overview', icon: BarChart3 },
              { id: 'transactions', label: 'Transactions', icon: Activity },
              { id: 'analytics', label: 'Analytics', icon: PieChart },
              { id: 'reports', label: 'Reports', icon: Download }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="px-6 py-8">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard
                title="Total Transactions"
                value={stats.totalTransactions.toLocaleString()}
                icon={Activity}
                color="blue"
              />
              <StatCard
                title="Fraud Detected"
                value={stats.fraudDetected}
                icon={AlertTriangle}
                color="red"
              />
              <StatCard
                title="Prevented Loss"
                value={`$${stats.preventedLoss.toLocaleString()}`}
                icon={DollarSign}
                color="green"
              />
              <StatCard
                title="Accuracy"
                value={`${stats.accuracy}%`}
                icon={TrendingUp}
                color="purple"
              />
            </div>

            {/* Quick Actions */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button className="flex items-center space-x-3 p-4 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors">
                  <Search className="w-5 h-5 text-blue-600" />
                  <span className="text-blue-700 font-medium">Search Transactions</span>
                </button>
                <button className="flex items-center space-x-3 p-4 bg-green-50 rounded-lg hover:bg-green-100 transition-colors">
                  <Download className="w-5 h-5 text-green-600" />
                  <span className="text-green-700 font-medium">Export Reports</span>
                </button>
                <button className="flex items-center space-x-3 p-4 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors">
                  <Settings className="w-5 h-5 text-purple-600" />
                  <span className="text-purple-700 font-medium">Configure Rules</span>
                </button>
              </div>
            </div>

            {/* Recent Transactions */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Recent High-Risk Transactions</h3>
              <div className="space-y-3">
                {transactions.filter(tx => tx.risk_level === 'High').slice(0, 5).map(tx => (
                  <div key={tx.transaction_id} className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                      <div>
                        <p className="font-medium text-gray-900">{tx.transaction_id}</p>
                        <p className="text-sm text-gray-500">${tx.amount.toLocaleString()}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-red-600">{(tx.fraud_score * 100).toFixed(1)}%</span>
                      <button
                        onClick={() => handleTransactionClick(tx)}
                        className="p-1 text-red-600 hover:bg-red-100 rounded"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'transactions' && (
          <div className="space-y-6">
            {/* Search and Filter */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search transactions..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <select
                  value={filterRisk}
                  onChange={(e) => setFilterRisk(e.target.value)}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="all">All Risk Levels</option>
                  <option value="high">High Risk</option>
                  <option value="medium">Medium Risk</option>
                  <option value="low">Low Risk</option>
                </select>
                <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2">
                  <RefreshCw className="w-4 h-4" />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Transactions Table */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Transaction ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Amount
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Sender
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Receiver
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Risk Level
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Fraud Score
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filteredTransactions.map((transaction) => (
                      <tr key={transaction.transaction_id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {transaction.transaction_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          ${transaction.amount.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {transaction.sender}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {transaction.receiver}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(transaction.risk_level)}`}>
                            {transaction.risk_level}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {(transaction.fraud_score * 100).toFixed(1)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {getStatusIcon(transaction.final_flag)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <button
                            onClick={() => handleTransactionClick(transaction)}
                            className="text-blue-600 hover:text-blue-900 mr-3"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button className="text-gray-600 hover:text-gray-900">
                            <Download className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Analytics Dashboard</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-gray-50 rounded-lg p-6 h-64 flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-500">Fraud Detection Trends</p>
                    <p className="text-xs text-gray-400">Chart visualization would go here</p>
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-6 h-64 flex items-center justify-center">
                  <div className="text-center">
                    <PieChart className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-500">Risk Level Distribution</p>
                    <p className="text-xs text-gray-400">Chart visualization would go here</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'reports' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Report Generation</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Report Type</label>
                    <select className="w-full p-2 border border-gray-300 rounded-lg">
                      <option>Daily Summary</option>
                      <option>Weekly Analysis</option>
                      <option>Monthly Report</option>
                      <option>Custom Range</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
                    <div className="flex space-x-2">
                      <input type="date" className="flex-1 p-2 border border-gray-300 rounded-lg" />
                      <input type="date" className="flex-1 p-2 border border-gray-300 rounded-lg" />
                    </div>
                  </div>
                  <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors">
                    Generate Report
                  </button>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium mb-2">Recent Reports</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-2 bg-white rounded">
                      <span className="text-sm">Daily Report - 2024-01-15</span>
                      <Download className="w-4 h-4 text-gray-500 cursor-pointer" />
                    </div>
                    <div className="flex items-center justify-between p-2 bg-white rounded">
                      <span className="text-sm">Weekly Analysis - Week 2</span>
                      <Download className="w-4 h-4 text-gray-500 cursor-pointer" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Transaction Modal */}
      {selectedTransaction && (
        <TransactionModal
          transaction={selectedTransaction}
          onClose={() => setSelectedTransaction(null)}
        />
      )}
    </div>
  );
};

export default Dashboard;