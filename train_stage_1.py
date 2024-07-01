from dependencies import *
from utils import *
from metric import MetricManager
from model import DilatedAutoEncoder, ModelManager
from dataset import CompleteDataset3DFixMask

random_seed = 42
setup_random_seed(seed=random_seed)

# cli
arg_parser = ArgumentParser()
# basic option
arg_parser.add_argument("--model_name", type=str, default="sas_model")
# dataset option
arg_parser.add_argument("--com_dataset_dir", type=str, default="../dataset/complete_dataset")
# save option
arg_parser.add_argument("--save_dir", type=str, default="../save_dir")
# train option
arg_parser.add_argument("--log_interval", type=int, default=1)
arg_parser.add_argument("--sample_interval", type=int, default=1)
arg_parser.add_argument("--save_interval", type=int, default=2)
arg_parser.add_argument("--batch_size", type=int, default=1)
# parse args
args = arg_parser.parse_args()

# paths
model_name = args.model_name
com_dataset_dir = args.com_dataset_dir
save_dir = args.save_dir
model_dir = f"{save_dir}/{model_name}"
log_dir = f"{model_dir}/log_dir"

# dataset dir
train_data_dir = f"{com_dataset_dir}/train/image/*"
val_data_dir = f"{com_dataset_dir}/val/image/*"
sample_data_dir = f"{com_dataset_dir}/sample/image/*"

# sample dir
train_sample_dir = f"{model_dir}/train_sample_dir"

# initialize dir
os.makedirs(model_dir, exist_ok=True)
os.makedirs(train_sample_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# log path
log_path = f"{log_dir}/log.txt"
init_log(log_path)
log(args)
log(f"[Start Time]:{get_cur_time_str()}")

# train params
step, cur_epoch = 0, 0
epoch_num = 50
batch_size = args.batch_size
log_interval = args.log_interval
save_interval = args.save_interval
sample_interval = args.sample_interval
learning_rate = 1e-4
fill_value = 1300/2300
eps = 1/128**3
# model params
num_hiddens = 768
num_residual_hiddens = 256
num_residual_layers = 3
embedding_dim = 768

# init model
model = DilatedAutoEncoder(in_channels=3,
                           num_hiddens=num_hiddens,
                           num_residual_layers=num_residual_layers,
                           num_residual_hiddens=num_residual_hiddens,
                           embedding_dim=embedding_dim)
opt_model = optim.Adam(
                        model.parameters(),
                        lr=learning_rate,
                        betas=(0.5,0.9),
                        eps=1e-8,
                        weight_decay=1e-5,
                        amsgrad=False
                       )

# load params
if(os.path.exists(f"{model_dir}/newest_ckpt.pth")):
    log("loading ckpt...")
    ckpt = torch.load(f"{model_dir}/newest_ckpt.pth")
    step, cur_epoch, learning_rate = ckpt["step"], ckpt["epoch"], ckpt["learning_rate"]
    cur_epoch = cur_epoch + 1
    log(f"loaded step:{step} epoch:{cur_epoch} learning_rate:{learning_rate}")
if(os.path.exists(f"{model_dir}/newest_model.pth")):
    log("loading autoencoder...")
    model_state = torch.load(f"{model_dir}/newest_model.pth")
    model.load_state_dict(model_state)
    log("unet3d loaded")
if(os.path.exists(f"{model_dir}/newest_opt_model.pth")):
    log("loading optim_model...")
    opt_model_state = torch.load(f"{model_dir}/newest_opt_model.pth")
    opt_model.load_state_dict(opt_model_state)

# gen dataset
# train
train_dataset = CompleteDataset3DFixMask(train_data_dir)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size)
train_datasize = len(train_dataloader)
# val
val_dataset = CompleteDataset3DFixMask(val_data_dir)
val_dataloader = DataLoader(dataset=val_dataset,batch_size=batch_size)
val_datasize = len(val_dataloader)
# sample
sample_dataset = CompleteDataset3DFixMask(sample_data_dir, ret_name=True)
sample_dataloader = DataLoader(dataset=sample_dataset,batch_size=1)
sample_datasize = len(sample_dataloader)

log(f"train_datasize:{train_datasize} val_datasize:{val_datasize} sample_datasize:{sample_datasize}")

# managers
model_manager=ModelManager([model])
metric_manager=MetricManager(root_dir=log_dir)

# one sample iter
def sample_gen(
    gen:DilatedAutoEncoder,
    x,
    mask,
    bony_border_mask,
    name
):
    with torch.no_grad():
        masked_x = x*(1-mask) + fill_value*mask
        input_x = torch.cat([masked_x, mask, bony_border_mask],dim=1)
        rec_x = gen.forward(input_x)
        con_rec_x = x*(1-mask) + rec_x*mask
        # image
        image_path = f"{train_sample_dir}/{name}_origin.nii.gz"
        write_mha_tensor4D(x[0], image_path)
        # masked_x image
        masked_image_path=f"{train_sample_dir}/{name}_{epoch}_masked.nii.gz"
        write_mha_tensor4D(masked_x[0], masked_image_path)
        # mask image
        mask_path = f"{train_sample_dir}/{name}_{epoch}_mask.nii.gz"
        write_mha_tensor4D(mask[0], mask_path)
        # bony_border_mask image
        bony_border_mask = f"{train_sample_dir}/{name}_{epoch}_bony_border_mask.nii.gz"
        write_mha_tensor4D(bony_border_mask[0], bony_border_mask)
        # output_x image
        rec_path = f"{train_sample_dir}/{name}_{epoch}_rec.nii.gz"
        write_mha_tensor4D(rec_x[0], rec_path)
        # con_rec_x image
        con_rec_path = f"{train_sample_dir}/{name}_{epoch}_con_rec.nii.gz"
        write_mha_tensor4D(con_rec_x[0], con_rec_path)
        #los loss
        log(f"sample name:{name} l1_loss:{torch.mean(torch.abs(rec_x-x))}")

# one train iter
def update_gen(
               gen:DilatedAutoEncoder,
               opt_gen:optim.Adam,
               x,
               mask,
               bony_border_mask,
               metric_manager,
               perfix=""):
    x_masked = x*(1-mask)+fill_value*mask
    input_x = torch.cat([x_masked, mask, bony_border_mask], dim=1)
    rec_x = gen.forward(input_x)
    # regular loss
    l1_loss_alpha = 1.0
    g_me_loss = l1_loss_alpha*torch.mean(torch.abs((x-rec_x)*mask))
    g_ae_loss = l1_loss_alpha*torch.mean(torch.abs((x-rec_x)*(1-mask)))
    metric_manager.update(f"gen/g_me_loss", g_me_loss, perfix)
    metric_manager.update(f"gen/g_ae_loss", g_ae_loss, perfix)
    metric_manager.update(f"gen/g_basic_loss", g_ae_loss+g_me_loss, perfix=perfix)
    g_loss = g_me_loss + g_ae_loss
    # sym loss
    sym_loss_alpha = 5
    inv_rec_x = get_register_inv_tensor(rec_x)
    g_sym_loss = sym_loss_alpha * torch.mean(torch.abs(inv_rec_x-rec_x))
    metric_manager.update("gen/g_sym_loss", g_sym_loss, perfix)
    g_loss += g_sym_loss
    # total loss
    metric_manager.update("gen/g_total_loss", g_loss, perfix)
    # optmize
    if(opt_gen is not None):
        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()
    #del
    del x, mask, bony_border_mask, x_masked, input_x, rec_x, inv_rec_x, g_loss


# train process
model_manager.train()
metric_manager.train()
model_manager.cuda()
for epoch in range(cur_epoch,epoch_num):
    log(f"epoch:{epoch}/{epoch_num}")
    start_time = time.time()

    # sample
    if(epoch%sample_interval==0):
        with torch.no_grad():
            for x, mask, bony_border_mask, name in tqdm(sample_dataloader):
                name = name[0]
                sample_gen(
                    gen = model,
                    x = x,
                    mask = mask,
                    bony_border_mask = bony_border_mask,
                    name = name
                )

    # train
    model_manager.train()
    metric_manager.train()
    for x, mask, bony_border_mask in tqdm(train_dataloader):
        step += 1
        # optimize gen
        model.set_requires_grad(True)
        update_gen(
                   gen=model,
                   opt_gen=opt_model,
                   x=x,
                   mask=mask,
                   bony_border_mask=bony_border_mask,
                   metric_manager=metric_manager)

        if(step%log_interval==0):
            log(f"[Train][{get_cur_time_str()}] epoch:{epoch} step:{step} learning_rate:{learning_rate} \n{metric_manager.report_log()}\n")
    
    train_end_time=time.time()

    # val
    model_manager.eval()
    metric_manager.eval()
    for x, mask, bony_border_mask in tqdm(val_dataloader):
        with torch.no_grad():
            update_gen(
                       gen=model,
                       opt_gen=None,
                       x=x,
                       mask=mask,
                       bony_border_mask=bony_border_mask,
                       metric_manager=metric_manager)

    val_end_time = time.time()
    log(f"[Train Summary][{get_cur_time_str()}] epoch:{epoch} step:{step} time:{(train_end_time-start_time)/3600} learning_rate:{learning_rate} \n{metric_manager.report_train()}\n")
    
    log(f"[Val Summary][{get_cur_time_str()}] epoch:{epoch} step:{step} time:{(val_end_time-train_end_time)/3600} learning_rate:{learning_rate} \n{metric_manager.report_val()}\n")
    
    # draw
    metric_manager.draw()
    
    # save
    ckpt = {}
    ckpt["step"], ckpt["epoch"], ckpt["learning_rate"] = step, epoch, learning_rate
    torch.save(ckpt, f"{model_dir}/newest_ckpt.pth")
    torch.save(model.state_dict(), f"{model_dir}/newest_model.pth")
    torch.save(opt_model.state_dict(), f"{model_dir}/newest_opt_model.pth")
    if(epoch%save_interval == 0):
        torch.save(ckpt, f"{model_dir}/{epoch}_ckpt.pth")
        torch.save(model.state_dict(), f"{model_dir}/{epoch}_model.pth")
        torch.save(opt_model.state_dict(), f"{model_dir}/{epoch}_opt_model.pth")
    log("ckpt saved!")