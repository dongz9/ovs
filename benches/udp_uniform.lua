local mg        = require "moongen"
local memory    = require "memory"
local device    = require "device"
local stats     = require "stats"
local timer     = require "timer"
local log       = require "log"

function configure(parser)
    parser:argument("txDev", "Device to transmit from."):convert(tonumber)
    parser:argument("rxDev", "Device to receive from."):convert(tonumber)
    parser:argument("timeLimit", "Time limit in seconds"):convert(tonumber)
    parser:argument("count", "Flow count per table"):convert(tonumber)
end

function master(args)
    math.randomseed(os.time())

    local numTxQueues = 1
	txDev = device.config{port = args.txDev, txQueues = numTxQueues}
    rxDev = device.config{port = args.rxDev, rxQueues = 1}
    device.waitForLinks()

    for i = 0, numTxQueues - 1 do
        mg.startTask("loadTask", txDev:getTxQueue(i), i, 60, args.count)
    end
    mg.startTask("counterTask", rxDev:getRxQueue(0))
    mg.setRuntime(args.timeLimit)
    mg.waitForTasks()
end

function div(x, y)
    return (x - x % y) / y
end

function loadTask(queue, port, size, count)
	local mempool = memory.createMemPool(function(buf)
		buf:getUdpPacket():fill{
            ethSrc = queue,
            ethDst = "90:E2:BA:B3:75:A1",
            ip4Src = "1.0.0.0",
            ip4Dst = "0.0.0.0",
            udpSrc = 1234,
            udpDst = 319,
            pktLength = size
        }
    end)
    local bufs = mempool:bufArray()
    local ctr = stats:newManualTxCounter("Port " .. port, "plain")
    while mg.running() do
        bufs:alloc(size)
        for i, buf in ipairs(bufs) do
            local x = math.random(20 * count) - 1
            local pkt = buf:getUdpPacket()
            pkt.ip4.src:set((div(x, count) + 1) * 16777216)
            pkt.ip4.dst:set(x % count)
        end
		-- UDP checksums are optional, so using just IPv4 checksums would be sufficient here
		bufs:offloadUdpChecksums()
		ctr:updateWithSize(queue:send(bufs), size)
    end
    ctr:finalize()
end

function counterTask(queue)
	local bufs = memory.bufArray()
    local ctr = stats:newPktRxCounter("RX", "plain")
	while mg.running() do
		local rx = queue:recv(bufs)
		for i = 1, rx do
			local buf = bufs[i]
			ctr:countPacket(buf)
        end
        ctr:update()
		bufs:freeAll()
    end
    ctr:finalize()
end