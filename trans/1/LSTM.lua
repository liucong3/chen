
require 'nn'
require 'Dynamic'
require 'MergeTable'
require 'BranchTable'

local LSTM, parent = torch.class('nn.LSTM', 'nn.Dynamic')

function LSTM:__init(hidden_size, input_size, W)
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.W = W or nn.Linear(hidden_size + input_size, 4 * hidden_size)
    parent.__init(self, self.W)
    self:options(true, true)
end

function LSTM:updateOutput(input)
	self.forwSequence = {}
    local prev_c, h, x = unpack(input)
    local h_x = self:DF(nn.MergeTable(2), h, x)
    local a = self:DF(self.W, h_x)
	local hs = self.hidden_size;
    local u, ifo = self:DF(nn.BranchTable{hs, 3 * hs}, a)
    u = self:DF(nn.Tanh(), u)
    ifo = self:DF(nn.Sigmoid(), ifo)
	local i, f, o = self:DF(nn.BranchTable{hs, hs, hs}, ifo)
    local c1 = self:DF(nn.CMulTable(), prev_c, f)
    local c2 = self:DF(nn.CMulTable(), u, i)
    local c = self:DF(nn.CAddTable(), c1, c2)
    local h = self:DF(nn.CMulTable(), c, o)
    return self:setOutput(c, h)
end

function LSTM:shareClone()
	local lstm = nn.LSTM(self.hidden_size, self.input_size, self.W)
	return lstm
end

-------

---[[
require 'gradientCheck'

local hidden_size = 3
local input_size = 4
lstm = nn.LSTM(hidden_size, input_size)
gradientCheck(lstm, {5, 2 * hidden_size + input_size}, {hidden_size, hidden_size, input_size})

--[[]]--
