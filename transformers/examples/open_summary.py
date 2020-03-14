from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt



def saveFig(tb, t1, t2, t3, pltn):



    fb, fnb = tb
    f1, fn1 = t1
    f2, fn2 = t2
    f3, fn3 =t3

    CoLA_baseline = EventAccumulator(fb)
    CoLA_on_MRPC = EventAccumulator(f1)
    CoLA_on_RTE = EventAccumulator(f2)
    CoLA_on_WNLI = EventAccumulator(f3)

    CoLA_baseline.Reload()
    CoLA_on_MRPC.Reload()
    CoLA_on_RTE.Reload()
    CoLA_on_WNLI.Reload()

    def getLoss(event):
        iterations = []
        losses = []
        for wall_time, iteration, loss in event.Scalars("loss"):
            iterations.append(iteration)
            losses.append(loss)
        return iterations, losses

    baseline = getLoss(CoLA_baseline)
    MRPC = getLoss(CoLA_on_MRPC)
    RTE = getLoss(CoLA_on_RTE)
    WNLI = getLoss(CoLA_on_WNLI)


    plt.clf()
    plt.plot(baseline[0], baseline[1], label=fnb)
    plt.plot(MRPC[0], MRPC[1], label=fn1)
    plt.plot(RTE[0], RTE[1], label=fn2)
    plt.plot(WNLI[0], WNLI[1], label=fn3)
    plt.legend(loc="upper right")
    plt.savefig(pltn + ".jpg")

saveFig(('runs/CoLA_baseline/Mar14_05-03-21_nv6-2/events.out.tfevents.1584162201.nv6-2.50230.0', "CoLA finetuned on Bert"), \
('runs/CoLA_on_MRPC/Mar14_09-17-48_nv6-2/events.out.tfevents.1584177468.nv6-2.12375.0', "CoLA finetuned on MRPC"), \
('runs/CoLA_on_RTE/events.out.tfevents.1584163392.nv6-2.56526.0', "CoLA finetuned on RTE"), \
('runs/CoLA_on_WNLI/Mar14_05-10-06_nv6-2/events.out.tfevents.1584162606.nv6-2.52783.0', "CoLA finetuned on WNLI"), \
 "CoLA"
)

saveFig(('runs/MRPC_on_CoLA/Mar14_04-49-48_nv6-2/events.out.tfevents.1584161388.nv6-2.45811.0', "MRPC fintuned on CoLA"), \
('runs/MRPC_baseline/Mar14_03-24-09_nv6-2/events.out.tfevents.1584156249.nv6-2.27890.0', "MRPC finetuend on BERT"), \
('runs/MRPC_on_RTE/Mar14_04-58-16_nv6-2/events.out.tfevents.1584161896.nv6-2.48616.0',"MRPC finetuned on RTE"), \
('runs/MRPC_on_WNLI/Mar14_04-53-55_nv6-2/events.out.tfevents.1584161635.nv6-2.47192.0', "MRPC finetuned on WNLI"), \
"MRPC"
)


saveFig(('runs/RTE_on_CoLA/events.out.tfevents.1584155646.nv6-2.25539.0', "RTE fintuned on CoLA"), \
('runs/RTE_on_MRPC/events.out.tfevents.1584155169.nv6-2.23149.0', "RTE finetuend on MRPC"), \
('runs/RTE_baseline/events.out.tfevents.1584153055.nv6-2.13290.0',"RTE finetuned on BERT"), \
('runs/RTE_on_WNLI/events.out.tfevents.1584155387.nv6-2.24269.0', "RTE finetuned on WNLI"), \
"RTE"
)

saveFig(('runs/WNLI_on_CoLA/Mar14_04-20-02_nv6-2/events.out.tfevents.1584159602.nv6-2.39657.0', "WNLI fintuned on CoLA"), \
('runs/WNLI_on_MRPC/events.out.tfevents.1584159888.nv6-2.40926.0', "WNLI finetuend on MRPC"), \
('runs/WNLI_on_RTE/events.out.tfevents.1584159765.nv6-2.40357.0',"WNLI finetuned on RTE"), \
('runs/WNLI_baseline/events.out.tfevents.1584212031.nv6-2.9116.0', "WNLI finetuned on BERT"), \
"WNLI"
)

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
#w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'