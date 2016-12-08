require 'nn'
require 'MergeTable'
require 'BranchTable'
require 'fixes'

---fixes-------------

function nn.Jacobian.testJacobianParameters(module, input, param, dparam, minval, maxval, perturbation)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   --input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   local jac_bprop = nn.Jacobian.backward(module, input, param, dparam)
   local jac_fprop = nn.Jacobian.forward(module, input, param, perturbation)
   local error = jac_fprop - jac_bprop
   return error:abs():max()
end

function nn.Jacobian.testJacobianUpdateParameters(module, input, param, minval, maxval, perturbation)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   --input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   local params_bprop = nn.Jacobian.backwardUpdate(module, input, param)
   local params_fprop = nn.Jacobian.forwardUpdate(module, input, param, perturbation)

   local error = params_fprop - params_bprop
   return error:abs():max()
end

--[[
Perform gradient check for an NLP with a single input and output,
with a random input matrix of size [batch_size x input_size]

s = nn.Sequential()
s:add(nn.Linear(3, 4))
s:add(nn.Tanh())
s:add(nn.Linear(4, 5))
s:add(nn.Sigmoid())
gradientCheck1(s, {10, 3})

nn.Jacobian.testJacobian	
6.676828534502e-11	
nn.Jacobian.testJacobianParameters #1	
1.1385822840104e-12	
nn.Jacobian.testJacobianUpdateParameters #1	
1.1834838664626e-12	
nn.Jacobian.testJacobianParameters #2	
4.339258258268e-10	
nn.Jacobian.testJacobianUpdateParameters #2	
4.3548531447613e-10	
nn.Jacobian.testJacobianParameters #3	
6.761929341112e-13	
nn.Jacobian.testJacobianUpdateParameters #3	
8.9006579884199e-13	
nn.Jacobian.testJacobianParameters #4	
2.087227612968e-10	
nn.Jacobian.testJacobianUpdateParameters #4	
2.089393658089e-10	

]]--

function _getInput(input, maxval, minval)
	if type(input) == 'table' and type(input[1]) == 'number' then
		input = torch.Tensor(input[1], input[2]) -- batch_size, input_size
		if maxval and minval then
			local inrange = maxval - minval
			input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
		end
	end
	return input
end

function gradientCheck1(mlp, input)
	local param, dparam = mlp:parameters()
	local minval = -0.1
	local maxval = 0.1
	local perturbation = 1e-4
	input = _getInput(input, maxval, minval)
	local output = mlp:forward(input)
	mlp:backward(input, output, 1)
	if type(mlp.gradInput) == 'table' then
		for i = 1, #mlp.gradInput do
			if #mlp.gradInput[i]:size() ~= 0 then
				print('nn.Jacobian.testJacobian')
				print(nn.Jacobian.testJacobian(mlp, input[i], minval, maxval, perturbation))
			end
		end
	else
		if #mlp.gradInput:size() ~= 0 then
			print('nn.Jacobian.testJacobian')
			print(nn.Jacobian.testJacobian(mlp, input, minval, maxval, perturbation))
		end
	end
	for i = 1, #param do
		if #dparam[i]:size() ~= 0 then
			print('nn.Jacobian.testJacobianParameters #' .. i)
			print(nn.Jacobian.testJacobianParameters(mlp, input, param[i], dparam[i], minval, maxval, perturbation))
		end
		print('nn.Jacobian.testJacobianUpdateParameters #' .. i)
		print(nn.Jacobian.testJacobianUpdateParameters(mlp, input, param[i], minval, maxval, perturbation))
	end
end

--[[
Perform gradient check for an NLP with multiple input and output,
with a random input matrix of sizes
 [batch_size x input_sizes(1)] ... [batch_size x input_sizes(#input_sizes)]

p = nn.ParallelTable()
p:add(nn.Linear(3,4))
p:add(nn.Linear(4,5))
gradientCheck(p, {5, 7}, {3, 4})

nn.Jacobian.testJacobian	
4.5952130989235e-13	
nn.Jacobian.testJacobianParameters #1	
4.7942205760876e-13	
nn.Jacobian.testJacobianUpdateParameters #1	
4.1637526759786e-13	
nn.Jacobian.testJacobianParameters #2	
2.8643754035329e-14	
nn.Jacobian.testJacobianUpdateParameters #2	
2.8643754035329e-14	
nn.Jacobian.testJacobianParameters #3	
2.5916768731093e-13	
nn.Jacobian.testJacobianUpdateParameters #3	
2.4691360067663e-13	
nn.Jacobian.testJacobianParameters #4	
2.8643754035329e-14	
nn.Jacobian.testJacobianUpdateParameters #4	
2.8643754035329e-14	

]]--


function gradientCheck(mlp, input, input_sizes)
	local seq = nn.Sequential();
	if input_sizes then
		seq:add(nn.BranchTable(input_sizes))
	end
	seq:add(mlp)
	_input = _getInput(input)
	local output = seq:forward(_input)
	if torch.type(output) == 'table' then
		seq:add(nn.MergeTable(#output))
	end
	gradientCheck1(seq, input)
end

-- gradientCheck(nn.MergeTable(3), 7, {3,4,5})
-- gradientCheck(nn.BranchTable{3,4,5}, 7, 12)


