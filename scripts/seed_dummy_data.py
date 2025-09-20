#!/usr/bin/env python3
from src.services.dummy_data.service import create_tables, seed_defaults, generate_addresses, generate_identifiers, generate_dates_financial


def main():
    print("Creating tables...")
    create_tables()
    print("Seeding default names...")
    seed_defaults()
    print("Generating addresses...")
    generate_addresses(200)
    print("Generating identifiers...")
    generate_identifiers(200)
    print("Generating dates/financial...")
    generate_dates_financial(200)
    print("Done.")


if __name__ == '__main__':
    main()
