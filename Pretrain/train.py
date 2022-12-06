import os, sys
import time
import torch
import torch.nn as nn
from model import prepare
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    pd = prepare()
    OUTPUT_DIR = pd["OUTPUT_DIR"]
    config, loader, model, logger, optimizer, scheduler \
        = pd["config"], pd["data_loader"], pd["model"], pd["logger"], pd["optimizer"], pd["scheduler"]
    try:
        optimizer.zero_grad()
        logger("Begin Training.")
        accumulate_loss = []
        accumulate_mlm_loss = []
        accumulate_ke_loss = []
        for epoch in range(config["epochs"]):
            # train
            loader.dataset.resample()
            model.train()
            for i, batch in enumerate(loader):
                mlm_loss, ke_loss = model(batch)
                loss = (mlm_loss + ke_loss) / config["grad_accum_step"]
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                accumulate_loss.append(loss.item())
                accumulate_mlm_loss.append(mlm_loss.item() / config["grad_accum_step"])
                accumulate_ke_loss.append(ke_loss.item() / config["grad_accum_step"])
                if (i + 1) % config["grad_accum_step"] == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                if (i + 1) % config["show_loss_step"] == 0:
                    mean_loss = sum(accumulate_loss) / len(accumulate_loss)
                    mean_mlm_loss = sum(accumulate_mlm_loss) / len(accumulate_mlm_loss)
                    mean_ke_loss = sum(accumulate_ke_loss) / len(accumulate_ke_loss)
                    logger("Epoch %d, step %d / %d, loss = %.4f (mlm: %.4f, ke: %.4f)" \
                        % (epoch + 1, i + 1, len(loader), mean_loss, mean_mlm_loss, mean_ke_loss))
                    accumulate_loss = []
                    accumulate_mlm_loss = []
                    accumulate_ke_loss = []
            if (epoch + 1) % config["save_model_step"] == 0:
                save_dir = os.path.join(OUTPUT_DIR, "saved_model_%d" % (epoch + 1))
                os.mkdir(save_dir)
                model.model.save_pretrained(save_dir)
                logger("Epoch %d, save model." % (epoch + 1))
        logger.fp.close()
    except KeyboardInterrupt:
        logger("Interrupted.")
        logger.fp.close()
        os.system("rm -rf %s" % OUTPUT_DIR)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        import traceback
        logger("Got exception.")
        logger.fp.close()
        print(traceback.format_exc())
        os.system("rm -rf %s" % OUTPUT_DIR)