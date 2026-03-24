import type { Product } from '../types'

interface Props {
  product: Product
}

export function ProductCard({ product }: Props) {
  const fmt = (v: number) =>
    new Intl.NumberFormat('vi-VN').format(v) + ' VND'

  const stars = Math.round(product.rating || 0)
  const starsStr = '★'.repeat(stars) + '☆'.repeat(5 - stars)

  const orig = product.original_price
  const hasDiscount = orig && orig > product.price

  return (
    <div className="product-card">
      {product.thumbnail_url && (
        <img
          src={product.thumbnail_url}
          alt={product.product_name}
          className="card-img"
          onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
        />
      )}
      <div className="card-body">
        <p className="card-name">{product.product_name}</p>
        <p className="card-price">
          {hasDiscount && <s className="orig-price">{fmt(orig!)}</s>}
          <span className="sale-price">{fmt(product.price)}</span>
          {hasDiscount && (
            <span className="discount-tag">
              -{Math.round((1 - product.price / orig!) * 100)}%
            </span>
          )}
        </p>
        <p className="card-rating">
          <span className="stars">{starsStr}</span>
          {' '}
          {product.rating?.toFixed(1)} ({product.review_count?.toLocaleString()} reviews)
        </p>
        <a
          href={product.url}
          target="_blank"
          rel="noopener noreferrer"
          className="card-btn"
        >
          Mua tren Tiki
        </a>
      </div>
    </div>
  )
}
