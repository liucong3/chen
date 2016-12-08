require 'nn'

local CrossEntropyCriterion2, parent = torch.class('nn.CrossEntropyCriterion2', 'nn.Criterion')

function CrossEntropyCriterion2:__init()
	parent.__init(self)
end

function CrossEntropyCriterion2:updateOutput(input, target)
    local out = target:gt(input:size(2))
    if out:sum() == 0 then 
        out = nil
    else
        local out2 = out:typeAs(target)
        target = target:clone()
        target:add(out2 - torch.cmul(out2, target))
    end

    -- o_i = exp(z_i) / \sum_j exp(z_j)
    -- l(y,o) = -log(o_y), 
    -- l(y,z) = log(\sum exp(z_j))) - z_y
    local z = input
    local y = target:view(-1,1)
    local max_z = z:max(2)
    z = z - max_z:repeatTensor(1, z:size(2))
    local exp_z = torch.exp(z)
    local sum_exp_z = exp_z:sum(2)
    local z_y = z:gather(2, y)
    local l = torch.log(sum_exp_z) - z_y

    if out then
        l:csub(torch.cmul(out:typeAs(l), l))
    end
    self.output = l:mean()

    -- d(y)/d(z_k) = o_k - \delta_ky 
    local o_k = torch.cdiv(exp_z, sum_exp_z:repeatTensor(1, z:size(2)))
    local delta_ky = z.new():resizeAs(z):zero():scatter(2, y, 1)

    self.gradInput = (o_k - delta_ky) / y:size(1)
    if out then
        local mask = (-out + 1):view(-1,1):expandAs(self.gradInput)
        self.gradInput:cmul(mask:typeAs(self.gradInput))
    end

	return self.output
end

function CrossEntropyCriterion2:updateGradInput(input, target)
	return self.gradInput
end

--[[
local c1 = nn.CrossEntropyCriterion()
local c2 = nn.CrossEntropyCriterion2()
local input = torch.rand(5,4)
local target = torch.LongTensor({1,5,3,8,2})
local out = target:gt(input:size(2)):nonzero():view(-1)
local target2 = target:clone()
target2:indexFill(1, out, 1)

print('input\n', input)
print('target\n', target)
print('target2\n', target2)
print(c1:forward(input, target2))
print(c1:backward(input, target2))
for i = 1, 5 do
	print(c2:forward(input, target2))
	print(c2:backward(input, target2))
	print(c2:forward(input, target))
	print(c2:backward(input, target))
end
--[[]]--
