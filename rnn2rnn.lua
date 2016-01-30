require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local BatchLoader = require 'util.BatchLoader'
local model_utils = require 'util.model_utils'
local Embed = require 'model.Embed'
local GRU = require 'model.GRU'

cmd = torch.CmdLine()
cmd:text()
cmd:text('RNN-RNN for Skip Thoughts')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','book_corpus_small/','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-seq_length',30,'number of timesteps to unroll for')
cmd:option('-dim_word',400,'word vector dimensionality')
cmd:option('-encoder_forw_rnn_size',200,'size of RNN encoder forward internal state')
cmd:option('-encoder_back_rnn_size',200,'size of RNN encoder backward internal state')
cmd:option('-encoder_num_layers',2,'number of layers in the RNN encoder')
cmd:option('-decoder_rnn_size',400,'size of RNN decoder internal state')
cmd:option('-decoder_num_layers',2,'number of layers in the RNN decoder')
cmd:option('-min_freq',10,'min frequency of words in vocabulary')
-- optimization
cmd:option('-learning_rate',0.001,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',5,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0.2,'dropout for regularization, used after each RNN hidden layer of encoder. 0 = no dropout')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',200,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir','cv', 'output directory where checkpoints get written')
cmd:option('-savefile','rnn2rnn','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val split for data, in fractions
local split_sizes = {math.max(0, 1 - opt.val_frac), opt.val_frac} 
-- the size of sentence vector is the sum of rnn unit in last layer
opt.vec_size = opt.encoder_forw_rnn_size + opt.encoder_back_rnn_size

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    if ok then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('Package cunn not found! If cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.min_freq)
local vocab_size = loader.vocab_size  -- the number of words in vocabulary
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- create the model, encoder and then decoder
print('creating model...')
protos = {}
protos.embed = Embed.embed(opt.dim_word, vocab_size)
protos.encoder_forw = GRU.gru(opt.dim_word, opt.encoder_forw_rnn_size, opt.encoder_num_layers, opt.dropout, 0, vocab_size)
protos.encoder_back = GRU.gru(opt.dim_word, opt.encoder_back_rnn_size, opt.encoder_num_layers, opt.dropout, 0, vocab_size)
protos.decoder_prev = GRU.gru(opt.dim_word, opt.decoder_rnn_size, opt.decoder_num_layers, opt.dropout, opt.vec_size, vocab_size)
protos.decoder_next = GRU.gru(opt.dim_word, opt.decoder_rnn_size, opt.decoder_num_layers, opt.dropout, opt.vec_size, vocab_size)
protos.criterion_prev = nn.ClassNLLCriterion()
protos.criterion_next = nn.ClassNLLCriterion()

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
embed_params, embed_grad_params = model_utils.combine_all_parameters(protos.embed)
encoder_forw_params, encoder_forw_grad_params = model_utils.combine_all_parameters(protos.encoder_forw)
encoder_back_params, encoder_back_grad_params = model_utils.combine_all_parameters(protos.encoder_back)
decoder_prev_params, decoder_prev_grad_params = model_utils.combine_all_parameters(protos.decoder_prev)
decoder_next_params, decoder_next_grad_params = model_utils.combine_all_parameters(protos.decoder_next)

-- initialization
if do_random_init then
    embed_params:uniform(-0.06, 0.06) -- small uniform numbers
    encoder_forw_params:uniform(-0.06, 0.06) -- small uniform numbers
    encoder_back_params:uniform(-0.06, 0.06) -- small uniform numbers
    decoder_prev_params:uniform(-0.06, 0.06) -- small uniform numbers
    decoder_next_params:uniform(-0.06, 0.06) -- small uniform numbers
end

local tol_params_num = embed_params:nElement() + encoder_forw_params:nElement() + encoder_back_params:nElement()
                    + decoder_prev_params:nElement() + decoder_next_params:nElement()
print('number of parameters in the model: ' .. tol_params_num)

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    if name ~= 'embed' then 
        clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
    else
        -- 3 clones for embeding layers: x, y_prev, y_next
        clones[name] = model_utils.clone_many_times(proto, 3, not proto.parameters)
    end 
end

-- the initial state of the hidden states
encoder_forw_init_state = {}
encoder_back_init_state = {}
decoder_prev_init_state = {}
decoder_next_init_state = {}
for L=1,opt.encoder_num_layers do
    local encoder_forw_h_init = torch.zeros(opt.batch_size, opt.encoder_forw_rnn_size)
    local encoder_back_h_init = torch.zeros(opt.batch_size, opt.encoder_back_rnn_size)
    if opt.gpuid >=0 then 
        encoder_forw_h_init = encoder_forw_h_init:cuda() 
        encoder_back_h_init = encoder_back_h_init:cuda() 
    end
    table.insert(encoder_forw_init_state, encoder_forw_h_init:clone())
    table.insert(encoder_back_init_state, encoder_back_h_init:clone())
end
for L=1,opt.decoder_num_layers do
    local decoder_prev_h_init = torch.zeros(opt.batch_size, opt.decoder_rnn_size)
    local decoder_next_h_init = torch.zeros(opt.batch_size, opt.decoder_rnn_size)
    if opt.gpuid >=0 then 
        decoder_prev_h_init = decoder_prev_h_init:cuda() 
        decoder_next_h_init = decoder_next_h_init:cuda() 
    end
    table.insert(decoder_prev_init_state, decoder_prev_h_init:clone())
    table.insert(decoder_next_init_state, decoder_next_h_init:clone())
end

-- preprocessing helper function
function prepro(x, y_prev, y_next)
    x = x:transpose(1, 2):contiguous() -- swap the axes for faster indexing
    y_prev = y_prev:transpose(1, 2):contiguous()
    y_next = y_next:transpose(1, 2):contiguous()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        x = x:float():cuda()
        y_prev = y_prev:float():cuda()
        y_next = y_next:float():cuda()
    end
    return x, y_prev, y_next
end

-- update params using rmsprop
function update_param(x, dfdx, config, state)
    -- clamp dfdx
    dfdx:clamp(-opt.grad_clip, opt.grad_clip)

    -- get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8

    -- initialize mean square values and square gradient storage
    if not state.m then
        state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
    end

    -- calculate new (leaky) mean squared values
    state.m:mul(alpha)
    state.m:addcmul(1.0-alpha, dfdx, dfdx)

    -- perform update
    state.tmp:sqrt(state.m):add(epsilon)
    x:addcdiv(-lr, dfdx, state.tmp)
end

-- a full forward & backward pass
function myfeval()
    ------------------ clear the gradients -------------
    embed_grad_params:zero()
    encoder_forw_grad_params:zero()
    encoder_back_grad_params:zero()
    decoder_prev_grad_params:zero()
    decoder_next_grad_params:zero()
    ------------------ get minibatch -------------------
    local x, y_prev, y_next = loader:next_batch(1)
    x, y_prev, y_next = prepro(x, y_prev, y_next)
    --------------- forward pass embeding layer ---------------
    clones.embed[1]:training()
    clones.embed[2]:training()
    clones.embed[3]:training()
    local embed_x = clones.embed[1]:forward(x) -- get the embedding of x
    local embed_y_prev = clones.embed[2]:forward(y_prev) -- get the embedding of y_prev
    local embed_y_next = clones.embed[3]:forward(y_next) -- get the embedding of y_next
    --------------- forward pass encoder ---------------
    local encoder_forw_state = {[0] = encoder_forw_init_state}
    local encoder_back_state = {[0] = encoder_back_init_state}
    for t=1,opt.seq_length do
        clones.encoder_forw[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.encoder_forw[t]:forward{embed_x[t], unpack(encoder_forw_state[t-1])}
        encoder_forw_state[t] = {}
        for i=1,#encoder_forw_init_state do table.insert(encoder_forw_state[t], lst[i]) end -- extract the state, without output
    end
    for t=1,opt.seq_length do
        clones.encoder_back[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.encoder_back[t]:forward{embed_x[opt.seq_length-t+1], unpack(encoder_back_state[t-1])}
        encoder_back_state[t] = {}
        for i=1,#encoder_back_init_state do table.insert(encoder_back_state[t], lst[i]) end -- extract the state, without output
    end
    local sent_vec = torch.cat(encoder_forw_state[opt.seq_length][1], encoder_back_state[opt.seq_length][1])
    --------------- forward pass decoder ---------------
    local decoder_prev_state = {[0] = decoder_prev_init_state}
    local decoder_next_state = {[0] = decoder_next_init_state}
    local predictions_prev = {}           -- softmax outputs
    local predictions_next = {}           -- softmax outputs
    local loss_prev = 0
    local loss_next = 0
    for t=1,opt.seq_length do
        clones.decoder_prev[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.decoder_prev[t]:forward{sent_vec, embed_y_prev[t], unpack(decoder_prev_state[t-1])}
        decoder_prev_state[t] = {}
        for i=1,#decoder_prev_init_state do table.insert(decoder_prev_state[t], lst[i]) end -- extract the state, without output
        predictions_prev[t] = lst[#lst] -- last element is the prediction
        -- the decoder always use current word to predict next word
        if t < opt.seq_length then
            loss_prev = loss_prev + clones.criterion_prev[t]:forward(predictions_prev[t], y_prev[t+1])
        else
            loss_prev = loss_prev + clones.criterion_prev[t]:forward(predictions_prev[t], y_prev[t])
        end
    end
    loss_prev = loss_prev / opt.seq_length
    for t=1,opt.seq_length do
        clones.decoder_next[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.decoder_next[t]:forward{sent_vec, embed_y_next[t], unpack(decoder_next_state[t-1])}
        decoder_next_state[t] = {}
        for i=1,#decoder_next_init_state do table.insert(decoder_next_state[t], lst[i]) end -- extract the state, without output
        predictions_next[t] = lst[#lst] -- last element is the prediction
        -- the decoder always use current word to predict next word
        if t < opt.seq_length then
            loss_next = loss_next + clones.criterion_next[t]:forward(predictions_next[t], y_next[t+1])
        else
            loss_next = loss_next + clones.criterion_next[t]:forward(predictions_next[t], y_next[t])
        end
    end
    loss_next = loss_next / opt.seq_length
    ------------------ backward pass decoder -------------------
    local dvec = sent_vec:clone():zero() -- gradients on sentence vector
    local dembed_x = embed_x:clone():zero() -- gradients on embed_x
    local dembed_y_prev = embed_y_prev:clone():zero() -- gradients on embed_y_prev
    local dembed_y_next = embed_y_next:clone():zero() -- gradients on embed_y_next
    if opt.gpuid >= 0 then
        dvec:float():cuda()
        dembed_x:float():cuda()
    end
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local d_decoder_prev_state = {[opt.seq_length] = clone_list(decoder_prev_init_state, true)} -- true also zeros the clones
    local d_decoder_next_state = {[opt.seq_length] = clone_list(decoder_next_init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t
        if t == opt.seq_length then
            doutput_t = clones.criterion_prev[t]:backward(predictions_prev[t], y_prev[t])
        else
            doutput_t = clones.criterion_prev[t]:backward(predictions_prev[t], y_prev[t+1])
        end
        table.insert(d_decoder_prev_state[t], doutput_t)
        -- backprop through rnn
        local dlst = clones.decoder_prev[t]:backward({sent_vec, embed_y_prev[t], unpack(decoder_prev_state[t-1])}, d_decoder_prev_state[t])
        d_decoder_prev_state[t-1] = {}
        for k,v in pairs(dlst) do
            -- k=1 is gradient on sentence vector
            if k == 1 then dvec:add(v) end 
            -- k=2 is gradient on embed_x end
            if k == 2 then dembed_y_prev[t]:add(v) end
            -- k>2 are gradients on rnn hidden state
            if k > 2 then d_decoder_prev_state[t-1][k-2] = v end
        end
    end
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t
        if t == opt.seq_length then
            doutput_t = clones.criterion_next[t]:backward(predictions_next[t], y_next[t])
        else
            doutput_t = clones.criterion_next[t]:backward(predictions_next[t], y_next[t+1])
        end
        table.insert(d_decoder_next_state[t], doutput_t)
        -- backprop through rnn
        local dlst = clones.decoder_next[t]:backward({sent_vec, embed_y_next[t], unpack(decoder_next_state[t-1])}, d_decoder_next_state[t])
        d_decoder_next_state[t-1] = {}
        for k,v in pairs(dlst) do
            -- k=1 is gradient on sentence vector
            if k == 1 then dvec:add(v) end 
            -- k=2 is gradient on embed_x
            if k == 2 then dembed_y_next[t]:add(v) end
            -- k>2 are gradients on rnn hidden state
            if k > 2 then d_decoder_next_state[t-1][k-2] = v end
        end
    end
    ------------------ backward pass encoder -------------------
    local d_encoder_forw_state = {[opt.seq_length] = clone_list(encoder_forw_init_state, true)} -- true also zeros the clones
    local d_encoder_back_state = {[opt.seq_length] = clone_list(encoder_back_init_state, true)} -- true also zeros the clones
    d_encoder_forw_state[opt.seq_length][#encoder_forw_init_state] = dvec[{{}, {1,opt.encoder_forw_rnn_size}}]:clone()
    d_encoder_back_state[opt.seq_length][#encoder_back_init_state] = dvec[{{}, {opt.encoder_forw_rnn_size+1,opt.vec_size}}]:clone()
    for t=opt.seq_length,1,-1 do
        local dlst = clones.encoder_forw[t]:backward({embed_x[t], unpack(encoder_forw_state[t-1])}, d_encoder_forw_state[t])
        d_encoder_forw_state[t-1] = {}
        for k,v in pairs(dlst) do
            -- k == 1 is gradient on embed_x
            if k == 1 then dembed_x[t]:add(v) end
            -- k>1 are gradients on rnn hidden state
            if k > 1 then d_encoder_forw_state[t-1][k-1] = v end
        end
    end
    for t=opt.seq_length,1,-1 do
        local dlst = clones.encoder_back[t]:backward({embed_x[opt.seq_length-t+1], unpack(encoder_back_state[t-1])}, d_encoder_back_state[t])
        d_encoder_back_state[t-1] = {}
        for k,v in pairs(dlst) do
            -- k == 1 is gradient on x, skip
            if k == 1 then dembed_x[t]:add(v) end
            -- k>1 are gradients on rnn hidden state
            if k > 1 then d_encoder_back_state[t-1][k-1] = v end
        end
    end
    ------------------ backward pass embedding layer -------------------
    clones.embed[1]:backward({x}, dembed_x)
    clones.embed[2]:backward({x}, dembed_y_prev)
    clones.embed[3]:backward({x}, dembed_y_next)
    -- update the grad_params
    update_param(embed_params, embed_grad_params, optim_state_embed)
    update_param(encoder_forw_params, encoder_forw_grad_params, optim_state_encoder_forw)
    update_param(encoder_back_params, encoder_back_grad_params, optim_state_encoder_back)
    update_param(decoder_prev_params, decoder_prev_grad_params, optim_state_decoder_prev)
    update_param(decoder_next_params, decoder_next_grad_params, optim_state_decoder_next)
    return torch.exp((loss_prev + loss_next) / 2)
end

function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end
    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y_prev, y_next = loader:next_batch(split_index)
        x, y_prev, y_next = prepro(x, y_prev, y_next)
        --------------- forward pass embeding layer ---------------
        clones.embed[1]:evaluate()
        clones.embed[2]:evaluate()
        clones.embed[3]:evaluate()
        local embed_x = clones.embed[1]:forward(x) -- get the embedding of x
        local embed_y_prev = clones.embed[2]:forward(y_prev) -- get the embedding of y_prev
        local embed_y_next = clones.embed[3]:forward(y_next) -- get the embedding of y_next
        --------------- forward pass encoder ---------------
        local encoder_forw_state = {[0] = encoder_forw_init_state}
        local encoder_back_state = {[0] = encoder_back_init_state}
        for t=1,opt.seq_length do
            clones.encoder_forw[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            local lst = clones.encoder_forw[t]:forward{embed_x[t], unpack(encoder_forw_state[t-1])}
            encoder_forw_state[t] = {}
            for i=1,#encoder_forw_init_state do table.insert(encoder_forw_state[t], lst[i]) end -- extract the state, without output
        end
        for t=1,opt.seq_length do
            clones.encoder_back[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            local lst = clones.encoder_back[t]:forward{embed_x[opt.seq_length-t+1], unpack(encoder_back_state[t-1])}
            encoder_back_state[t] = {}
            for i=1,#encoder_back_init_state do table.insert(encoder_back_state[t], lst[i]) end -- extract the state, without output
        end
        local sent_vec = torch.cat(encoder_forw_state[opt.seq_length][1], encoder_back_state[opt.seq_length][1])
        --------------- forward pass decoder ---------------
        local decoder_prev_state = {[0] = decoder_prev_init_state}
        local decoder_next_state = {[0] = decoder_next_init_state}
        local predictions_prev = {}           -- softmax outputs
        local predictions_next = {}           -- softmax outputs
        local loss_prev = 0
        local loss_next = 0
        for t=1,opt.seq_length do
            clones.decoder_prev[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            local lst = clones.decoder_prev[t]:forward{sent_vec, embed_y_prev[t], unpack(decoder_prev_state[t-1])}
            decoder_prev_state[t] = {}
            for i=1,#decoder_prev_init_state do table.insert(decoder_prev_state[t], lst[i]) end -- extract the state, without output
            predictions_prev[t] = lst[#lst] -- last element is the prediction
            if t < opt.seq_length then
                loss_prev = loss_prev + clones.criterion_prev[t]:forward(predictions_prev[t], y_prev[t+1])
            else
                loss_prev = loss_prev + clones.criterion_prev[t]:forward(predictions_prev[t], y_prev[t])
            end
        end
        loss_prev = loss_prev / opt.seq_length
        for t=1,opt.seq_length do
            clones.decoder_next[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            local lst = clones.decoder_next[t]:forward{sent_vec, embed_y_next[t], unpack(decoder_next_state[t-1])}
            decoder_next_state[t] = {}
            for i=1,#decoder_next_init_state do table.insert(decoder_next_state[t], lst[i]) end -- extract the state, without output
            predictions_next[t] = lst[#lst] -- last element is the prediction
            if t < opt.seq_length then
                loss_next = loss_next + clones.criterion_next[t]:forward(predictions_next[t], y_next[t+1])
            else
                loss_next = loss_next + clones.criterion_next[t]:forward(predictions_next[t], y_next[t])
            end
        end
        loss_next = loss_next / opt.seq_length
        loss = loss + (loss_prev + loss_next) / 2
    end
    return torch.exp(loss/n)
end

-- optim state for emsprop
optim_state_embed = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
optim_state_encoder_forw = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
optim_state_encoder_back = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
optim_state_decoder_prev = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
optim_state_decoder_next = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
-- start optimization here
local loss0 = nil
train_losses = {}
val_losses = {}
local iterations = opt.max_epochs * loader.ntrain
for i = 1, iterations do
    local epoch = i / loader.ntrain
    local timer = torch.Timer()

    local train_loss = myfeval()
    train_losses[i] = train_loss
    local time = timer:time().real

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state_embed.learningRate = optim_state_embed.learningRate * decay_factor -- decay it
            optim_state_encoder_forw.learningRate = optim_state_encoder_forw.learningRate * decay_factor -- decay it
            optim_state_encoder_back.learningRate = optim_state_encoder_back.learningRate * decay_factor -- decay it
            optim_state_decoder_prev.learningRate = optim_state_decoder_prev.learningRate * decay_factor -- decay it
            optim_state_decoder_next.learningRate = optim_state_decoder_next.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. opt.learning_rate_decay .. ' to ' .. optim_state_embed.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        grad_norm = embed_grad_params:norm() / embed_params:norm()
        grad_norm = grad_norm + encoder_forw_grad_params:norm() / encoder_forw_params:norm()
        grad_norm = grad_norm + encoder_back_grad_params:norm() / encoder_back_params:norm()
        grad_norm = grad_norm + decoder_prev_grad_params:norm() / decoder_prev_params:norm()
        grad_norm = grad_norm + decoder_next_grad_params:norm() / decoder_next_params:norm()
        grad_norm = grad_norm / 4;
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_norm, time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss0 == nil then loss0 = train_loss end
    if train_loss > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end

