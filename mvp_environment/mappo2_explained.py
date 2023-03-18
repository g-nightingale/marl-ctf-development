
def training_loop():
    """Train the model on the given environment using multiple actors acting up to n timestamps."""


    print(f'\t##### Training loop ')
    print(f'\tInit wandb')

    # Training loop
    for iteration in range(3):
        print(f'\t\t##### Iteration {iteration}')

        for a in range(3):
            print(f'\t\t\tGet timetamps for actor: {a}')

        print(f'\t\tNormalise values and shuffle buffer')

        # Running optimization for a few epochs
        print(f'\t\t##### Optimise over epochs')
        for e in range(3):
            print(f'\t\t\tEpoch: {e}')
            for b in range(3):
                print(f'\t\t\t\tBatch: {b}')
                print(f'\t\t\t\tCalculate batch losses')
                print(f'\t\t\t\tOptimise')

        print(f'\t\tLog info to console and WANDB')

def main():
    print(f'##### Setup ')
    print(f'Get args')
    print(f'Define environment')
    print(f'Create actors and critic models')

    # Training
    training_loop()

    print(f'##### Finish ')


if __name__ == '__main__':
    main()