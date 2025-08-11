# pointnet with augmentation
for i in $(seq 1 5);
do
    sh scripts/train.sh -g 1 -d gim3d -c pointnet-t$i-aug -n pointnet-t$i-aug;
done

# dgcnn with augmentation
for i in $(seq 1 5);
do
    sh scripts/train.sh -g 1 -d gim3d -c closenet-t$i-aug -n closenet-t$i-aug;
done

# ptv1 with augmentation
for i in $(seq 1 5);
do
    sh scripts/train.sh -g 1 -d gim3d -c pt-t$i-aug -n pt-t$i-aug;
done